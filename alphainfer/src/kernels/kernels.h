// SPDX-License-Identifier: Apache-2.0
// AlphaInfer native kernels and stream-capture primitives (CUDA TUs in src/kernels).
//
// Plain C++ interface: no CUDA or ggml types. Every kernel is launched on the stream the ggml
// adapters also use, so native kernels and adapter Ops interleave in one captured round.
#pragma once

#include <cstdint>

struct ggml_backend;

namespace ai::kern {

// The compute stream of the ggml CUDA backend.
void * backend_stream(ggml_backend * backend);
// AI_MOE_CONCURRENT: pick which of the backend's own streams the next graph launches on (stream_fork.cu).
void   backend_set_stream_no(ggml_backend * backend, int n);
void * backend_stream_no(ggml_backend * backend, int n);

// Stream capture of a whole round: begin -> launches -> end (instantiates) -> launch many times.
struct Capture {
    void * graph = nullptr;
    void * exec  = nullptr;
};
bool capture_begin(void * stream);
bool capture_end(void * stream, Capture & c);
bool capture_launch(const Capture & c, void * stream);
void capture_free(Capture & c);

// y = res + ((sum_j exps[.., j, ..] * w[j]) + shared), summed in expert order j = 0..k-1.
// exps: f32 [E, k, T] contiguous; w: f32 [1, k, T]; shared, res, y: f32 [E, T].
// Bitwise equal to MUL, k-1 ADDs, ADD(shared), ADD(residual) issued as separate elementwise ops.
// Gated: shared = shared_down[i] * sigmoid(score[t*score_stride + score_off]) computed in the kernel.
void moe_merge_residual_gated(void * stream, const float * exps, const float * w, const float * shared_down,
                              const float * score, int64_t score_stride, int64_t score_off, const float * res,
                              float * y, int64_t E, int64_t k, int64_t T);
void moe_merge_residual(void * stream, const float * exps, const float * w, const float * shared,
                        const float * res, float * y, int64_t E, int64_t k, int64_t T);
// Same, plus the next layer's input RMS norm: out = rms_norm(y) * norm_w. One block per row.
// Cooperative variant: keeps the merge spread across the grid and pays two grid-wide barriers for
// the reduction, instead of collapsing to one block per row. Returns false if the launch is
// refused (cooperative launches are not permitted during stream capture on every driver), and the
// caller must then fall back.
bool moe_merge_residual_gated_norm_coop(void * stream, const float * exps, const float * w,
                                        const float * shared_down, const float * score,
                                        int64_t score_stride, int64_t score_off, const float * res,
                                        float * y, float * out, const float * norm_w,
                                        float * scale_scratch, int64_t E, int64_t k, int64_t T, float eps);
void moe_merge_residual_gated_norm(void * stream, const float * exps, const float * w,
                                   const float * shared_down, const float * score, int64_t score_stride,
                                   int64_t score_off, const float * res, float * y, float * out,
                                   const float * norm_w, int64_t E, int64_t k, int64_t T, float eps);
// Same fusion with the merge's full grid kept: the last block to finish norms every row (one atomic,
// no grid barrier). AI_FUSE_MERGENORM=3. Bit-exact against moe_merge_gated_k + rms_norm_f32<1024>.
void moe_merge_residual_gated_norm_last(void * stream, const float * exps, const float * w,
                                        const float * shared_down, const float * score, int64_t score_stride,
                                        int64_t score_off, const float * res, float * y, float * out,
                                        const float * norm_w, int64_t E, int64_t k, int64_t T, float eps);

// AI_FUSE_MERGENORM=4: the merge also writes the reference norm's 32 per-warp partial sums of squares per
// row; the apply kernel (the quantise's shape) finishes the norm and optionally emits the q8_1 copy.
bool moe_merge_partials_ok(int64_t E);
void moe_merge_residual_gated_partials(void * stream, const float * exps, const float * w,
                                       const float * shared_down, const float * score, int64_t score_stride,
                                       int64_t score_off, const float * res, float * y, float * partials,
                                       int64_t E, int64_t k, int64_t T, bool interleave);
void rms_apply_partials_q8(void * stream, const float * y, const float * partials, const float * norm_w,
                           float * out, void * q8, int64_t E, int64_t T, float eps);
// AI_ADDNORM_SPLIT: add_rmsnorm as a full-grid add + partials, then the apply. Returns false (and launches
// nothing) for shapes it does not cover; the caller then runs add_rmsnorm.
bool add_rmsnorm_split(void * stream, const float * mix, const float * res, const float * w, float * r, float * p,
                       float * partials, void * q8, int64_t E, int64_t T, float eps);
// AI_ADDNORM_SPLIT=2: one full-grid kernel, each block redundantly reduces the whole row (bit-exact).
bool add_rmsnorm_redundant(void * stream, const float * mix, const float * res, const float * w, float * r,
                           float * p, void * q8, int64_t E, int64_t T, float eps);
// AI_ADDNORM_ROUTER: add_rmsnorm_redundant + the F16 router GEMV (router_w [E, nrows]) in one launch; logits
// f32 [nrows, T]. Bit-exact with add_rmsnorm_redundant followed by ggml's mul_mat_vec_f<half, half>.
bool addnorm_router(void * stream, const float * mix, const float * res, const float * w, float * r, float * p,
                    void * q8, const void * router_w, int64_t nrows, float * logits, int64_t E, int64_t T, float eps);
void addnorm_router_compare(void * stream, const float * a, const float * b, int64_t n);
// AI_MERGE_APPLY: moe_merge_residual_gated_partials + rms_apply_partials_q8 in one launch (each block recomputes
// the merged row for the norm's partials). Bit-exact with the pair. E must be 2048.
bool moe_merge_apply_redundant(void * stream, const float * exps, const float * w, const float * shared_down,
                               const float * score, int64_t score_stride, int64_t score_off, const float * res,
                               float * y, const float * norm_w, float * out, void * q8, int64_t E, int64_t k, int64_t T,
                               float eps);

// AI_GDN_GATE: fin = silu(z) * (rms_norm(out) * w) for the GDN output gate, plus its q8_1 copy (q8 may be
// null). out/z strides are per token in floats; fin is contiguous [Sv*Hv, T]. Bit-exact with the ggml chain.
bool gdn_gated_norm_q8(void * stream, const float * out, int64_t out_tok, const float * z, int64_t z_tok,
                       const float * norm_w, float * fin, void * q8, int64_t Sv, int64_t Hv, int64_t T, float eps);
// AI_SMALL_GEMV: y[N, T] = W[N, K] (F16, row-major) x[K, T] (f32), f32 accumulate, T <= 8. For the small
// router|score and GDN beta|alpha spans. Returns false (launches nothing) for unsupported shapes/alignment.
bool gemv_f16w(void * stream, const void * W, const float * x, float * y, int64_t K, int64_t N, int64_t T,
               int64_t x_tok, int64_t y_tok);
// AI_ROUTER_TOPK: router GEMV (F16 W [nrows, K], half accumulate as ggml's mul_mat_vec_f) + topk_moe
// (softmax, top-k, clamp-normalise) in one launch; bit-exact with the pair. logits f32 [nrows, T], ids i32
// [k, T], weights f32 [k, T]. n_expert must be 256 and k 8; returns false otherwise.
bool router_topk(void * stream, const void * W, const float * y, float * logits, int32_t * ids, float * weights,
                 int64_t K_in, int64_t nrows, int64_t n_expert, int64_t k_used, int64_t T, int64_t y_tok, float clamp_val);
void router_topk_compare(void * stream, const float * a_l, const float * b_l, int64_t b_l_tok, int64_t nl,
                         const int32_t * a_i, const int32_t * b_i, int64_t b_i_tok, const float * a_w, const float * b_w,
                         int64_t k, int64_t T);
// AI_MOE_GU: MTP-verify routed-expert gate/up + SwiGLU in one launch, expert dedup + x reuse (moe_gu_verify.cu).
// x_q8: q8_1 copy of the [K, T] activation; ids [8, T] with row stride ids_stride; act f32 [F, 8, T].
// Bit-exact with mul_mat_vec_q_moe (gate), (up) + unary_gated_op_kernel<op_silu>. T 2..4, Q4_K/Q5_K.
bool moe_gu_verify_ok(int ggml_type_id, int64_t K, int64_t F, int64_t n_used, int64_t T);
bool moe_gu_verify(void * stream, int ggml_type_id, const void * up, const void * gate, int64_t row_bytes,
                   int64_t expert_bytes, const void * x_q8, const int32_t * ids, int64_t ids_stride, float * act,
                   int64_t K, int64_t F, int64_t T,
                   void * act_q8 = nullptr);
void moe_gu_compare(void * stream, const float * mine, const float * ref, int64_t n, int64_t T);
// AI_MOE_DN: MTP-verify routed-expert down projection, expert dedup + weight-unpack reuse (moe_gu_verify.cu).
// act_q8: q8_1 copy of act f32 [F, 8, T] (rows of F); ids as above; out f32 [N, 8, T]. Bit-exact with ggml's
// ids MMVQ. T 2..8, Q4_K/Q5_K, F a multiple of 256.
bool moe_down_verify_ok(int ggml_type_id, int64_t F, int64_t N, int64_t n_used, int64_t T);
bool moe_down_verify(void * stream, int ggml_type_id, const void * down, int64_t row_bytes, int64_t expert_bytes,
                     const void * act_q8, const int32_t * ids, int64_t ids_stride, float * out, int64_t F, int64_t N,
                     int64_t T);
void moe_down_compare(void * stream, const float * mine, const float * ref, int64_t n);
// AI_MOE_PF: prefill-sized routed experts as grouped int8 tensor-core GEMMs (moe_pf.cu). q4_K / q5_K experts.
// moe_pf_gu: x f32 [K, T] (column stride x_col_stride floats), ids [n_used, T] (row stride ids_stride) -> act f32
// [F, n_used, T] = SwiGLU(gate, up); also fills `scratch` (moe_pf_scratch_bytes) with the routing tables and the int8
// copy of act that moe_pf_down then reads (call it with the same scratch). out f32 [N_down, n_used, T].
// Not bit-exact with ggml's MMQ.
bool   moe_pf_ok(int gu_type, int64_t K, int64_t F, int64_t n_used, int64_t n_expert, int64_t T);
bool   moe_pf_down_ok(int down_type, int64_t K_down, int64_t N_down);
size_t moe_pf_scratch_bytes(int64_t K, int64_t F, int64_t n_used, int64_t n_expert, int64_t T);
bool   moe_pf_gu(void * stream, int type, const void * up, const void * gate, int64_t row_bytes, int64_t expert_bytes,
                 const float * x, int64_t x_col_stride, const int32_t * ids, int64_t ids_stride, float * act,
                 int64_t K, int64_t F, int64_t n_used, int64_t n_expert, int64_t T, void * scratch);
bool   moe_pf_down(void * stream, int type, const void * down, int64_t row_bytes, int64_t expert_bytes, float * out,
                   int64_t K, int64_t F, int64_t N_down, int64_t n_used, int64_t n_expert, int64_t T, void * scratch);
// AI_MTP_GATE: out[r] = max-softmax of row r of x f32 [ncols, rows] (a draft's confidence)
// scratch: max_softmax_scratch_floats(rows) floats of workspace (8-byte aligned), or nullptr for the single-block kernel
int64_t max_softmax_scratch_floats(int64_t rows);
void    max_softmax_rows(void * stream, const float * x, int64_t ncols, int64_t rows, float * out, float * scratch);
// AI_MOE_TC: verify gate/up + SwiGLU on tensor cores, T 2..8, owner dedup; not bit-exact (fast verify only).
// x f32 [K, T] with column stride x_col_stride floats; scratch = moe_gu_tc_scratch_bytes(K, T) bytes of device memory.
bool   moe_gu_tc_ok(int ggml_type_id, int64_t K, int64_t F, int64_t n_used, int64_t T);
size_t moe_gu_tc_scratch_bytes(int64_t K, int64_t T);
bool   moe_gu_tc(void * stream, int ggml_type_id, const void * up, const void * gate, int64_t row_bytes,
                 int64_t expert_bytes, const float * x, int64_t x_col_stride, void * scratch, const int32_t * ids,
                 int64_t ids_stride, float * act, int64_t K, int64_t F, int64_t T);
// AI_TOPK_NATIVE: softmax + top-8 + clamp-normalise from router logits (row stride nrows), bit-exact with
// ggml's topk_moe_cuda. ids i32 [8, T], w f32 [8, T].
bool topk_native_ok(int64_t n_expert, int64_t k_used, int64_t T);
// Compute-capability major of the current device (8 = Ampere/Ada, 12 = Blackwell consumer); 0 if unknown.
// For per-architecture defaults.
int device_cc_major();

// Offload v2 (offload.cu): device residency table + mapped-memory doorbell to the CPU miss pool.
// table: i32 [n_expert] expert -> pool slot or -1; sel: i32 [k, T] routed experts; x: f32 [N, T]. Writes slots
// (i32 [k, T], misses -> filler slots filler0 + j), and to the mapped bell: miss ids (i32 [k, T], -1 = hit), x, the
// token count, then seq = *base + layer + 1. Capturable: no host synchronisation.
void offload_resolve_ring(void * stream, const int32_t * table, const int32_t * sel, const float * x, int64_t N, int k,
                          int64_t T, int32_t filler0, int32_t * slots, int32_t * miss_bell, float * x_bell,
                          int32_t * n_tok_bell, unsigned * seq_bell, const unsigned * base, int layer,
                          const float * w = nullptr, float * w_bell = nullptr,   // w: routing weights [k*T] -> bell
                          int gpu_m = 0, int32_t gpu0 = INT32_MAX, int tail_rank = 1 << 30);   // GPU miss share
// AI_Q4X_PARTIAL: spin for the CPU, zero each token's missed columns of exps and add the CPU's per-token weighted sum
// yp (f32 [N, T], mapped) into shared (f32 [N, T]).
void offload_wait_partial(void * stream, const unsigned * done_bell, const unsigned * base, int layer,
                          const int32_t * slots, int32_t filler0, int32_t gpu0, const float * yp_bell, float * exps,
                          float * shared, int64_t N, int k, int64_t T);   // slots: >= filler0 = missed, >= gpu0 = GPU-taken
// Spin until the CPU answered this window's ring (done >= *base + layer + 1), then copy its rows y (f32 [N, k*T],
// column t*k + j) over the missed columns of exps (f32 [N, k, T], the unweighted expert outputs).
void offload_wait_fill(void * stream, const unsigned * done_bell, const unsigned * base, int layer,
                       const int32_t * miss_bell, const float * y_bell, float * exps, int64_t N, int k, int64_t T);
// Fused qwen4exp routing for T <= 16 (src/kernels/q4x_route.cu): router GEMV from BF16 (w_is_bf16) or F16 weights
// [E rows of K], softmax over E, top-k (ties -> lower index), weights renormalised over the top-k (min 6.1e-5).
// logits f32 [E, T] scratch; sel i32 [k, T]; weights f32 [k, T]. false = shape not supported (caller falls back).
bool q4x_route(void * stream, const void * w, bool w_is_bf16, const float * x, int K, int E, int T, int k,
               float * logits, int32_t * sel, float * weights);
// select_cols: out[:, c] = use_a[c] ? a[:, c] : b[:, c] over f32 [N, cols] (offload.cu)
void select_cols(void * stream, const float * a, const float * b, const int32_t * use_a, float * out, int64_t N,
                 int64_t cols);
// moe_wsum: fused MoE weighted sum, out f32 [N, T] = sum_j w[j, t] * exps[:, j, t] + shared[:, t], ggml's op order.
void moe_wsum(void * stream, const float * exps, const float * w, const float * shared, float * out, int64_t N, int k,
              int64_t T);
// qwen4exp routed experts grouped by slot (src/kernels/q4x_experts.cu): the lowest column routed to a slot owns
// it and dots each unpacked weight block against every column sharing it; columns on a filler slot (>= filler0, missed)
// are skipped. out f32 [N, k, T] (missed columns unwritten). Bit-exact with the ggml mul_mat_id + swiglu chain.
bool   q4x_experts_ok(int ty_gu, int ty_down, int64_t K, int64_t F, int64_t N, int k, int64_t T);
// GPU miss share: the same grouped experts over the columns whose ids lie in [id_lo, id_hi): expert id - id_lo of a
// host bank (the mapped RAM tier's device pointers), always on the smem-staged kernels (bulk 16-byte cp.async reads
// get the most out of the PCIe link). Writes only those columns of out.
bool   q4x_experts_range(void * stream, int ty_gu, const void * up, const void * gate, int64_t row_bytes_gu,
                         int64_t expert_bytes_gu, int ty_down, const void * down, int64_t row_bytes_down,
                         int64_t expert_bytes_down, const float * x, int64_t x_stride, const int32_t * ids,
                         int64_t ids_stride, int32_t id_lo, int32_t id_hi, int64_t K, int64_t F, int64_t N, int k,
                         int64_t T, void * scratch, float * out);
size_t q4x_experts_scratch_bytes(int64_t K, int64_t F, int k, int64_t T);
bool   q4x_experts(void * stream, int ty_gu, const void * up, const void * gate, int64_t row_bytes_gu,
                   int64_t expert_bytes_gu, int ty_down, const void * down, int64_t row_bytes_down,
                   int64_t expert_bytes_down, const float * x, int64_t x_stride, const int32_t * ids, int64_t ids_stride,
                   int filler0, int64_t K, int64_t F, int64_t N, int k, int64_t T, void * scratch, float * out);
// q4x_experts_compare (AI_Q4X_GROUPED=2): count bitwise differences between two [N, k, T] expert outputs over the non-missed columns
// q4x_experts_wide: the grouped experts over wide prefill windows (T <= q4x_experts_wide_cols()) with a host-built owner plan:
// n_own records of q4x_experts_wide_rec() ints, [slot, ncols, cols...] (column = t * k + j). Writes only owned columns.
// q4x_gemv16: small BF16/F16 projection of T <= 16 rows with the router's GEMV (q4x_route.cu): out[t * N + n] = W[n] . x[t];
// K % 512 == 0, K <= 4096, 16-byte aligned w and x. Returns false (nothing launched) otherwise.
// Draft capture: tok = map ? map[*idx] : *idx -> *ids_next (if non-null), *out_tok; *out_p = *p (q4x_route.cu)
void   q4x_draft_next(void * stream, const int32_t * idx, const float * p, const int32_t * map, int32_t * ids_next,
                      int32_t * out_tok, float * out_p);
// Prefill: one sub-chunk's causal mask f16 [n_kv, T]: m[i * n_kv + j] = j <= p0 + i ? 0 : -inf (q4x_route.cu)
void   q4x_causal_mask(void * stream, uint16_t * m, int64_t n_kv, int64_t T, int64_t p0);
void   q4x_mtp_mask_col0(void * stream, uint16_t * m, int64_t n_kv, int64_t T, int64_t p0);   // MTP: column 0 masked past p 0
// copy n i32 from (mapped pinned host) src to device dst with a kernel on `stream` (not the copy engine)
void   q4x_copy_i32(void * stream, const int32_t * src, int32_t * dst, int64_t n);
bool   q4x_gemv16(void * stream, const void * w, bool w_is_bf16, const float * x, int K, int N, int T, float * out);
// Dense small-M GEMV on fp16 tensor cores, fp16 activations (src/kernels/q4x_dense.cu): out[t * ldo + n] =
// sum_k W[n][k] x[t * x_stride + k] for q6_K / q5_K / q4_K / iq4_xs / iq3_s rows, T <= 32; flat in T per 8 columns.
bool   q4x_dense_ok(int type, int64_t N, int64_t K, int64_t T);
size_t q4x_dense_scratch_bytes(int64_t K, int64_t T);
bool   q4x_dense(void * stream, int type, const void * w, int64_t row_bytes, int64_t N, int64_t K, const float * x,
                 int64_t x_stride, int64_t T, void * scratch, float * out, int64_t ldo);
// 1 = the tensor-core grouped experts, 0 = the dp4a ones; overrides AI_Q4X_TC (for tools)
void   q4x_experts_set_tc(int mode);
int    q4x_experts_wide_rec();
// Prefill with big MoE batches: the tensor-core wide path in three steps (q4x_experts.cu): gate/up over T rows (both banks
// into one act region; quantize_x on the first), the act quantize, then the down projection for a set of owner records
// whose columns all lie in [col_lo, col_lo + ...), written to out + (col - col_lo) * N. Same arithmetic per column as
// q4x_experts_wide. split_ok: the shapes / types / mode take this path (else use q4x_experts_wide).
bool   q4x_experts_wide_split_ok(int ty_gu, const void * up, const void * gate, int64_t rbg, int64_t ebg, int ty_down,
                                 const void * down, int64_t rbd, int64_t ebd, int64_t K, int64_t F, int64_t N, int64_t T);
bool   q4x_experts_wide_gu(void * stream, int ty_gu, const void * up, const void * gate, int64_t rbg, int64_t ebg,
                           int ty_down, const void * down, int64_t rbd, int64_t ebd, const float * x, int64_t x_stride,
                           bool quantize_x, const int32_t * plan, int n_own, int64_t K, int64_t F, int64_t N, int k, int64_t T,
                           void * scratch);
void   q4x_experts_wide_qact(void * stream, int64_t K, int64_t F, int k, int64_t T, void * scratch);
bool   q4x_experts_wide_dn(void * stream, int ty_gu, const void * up, const void * gate, int64_t rbg, int64_t ebg,
                           int ty_down, const void * down, int64_t rbd, int64_t ebd, const int32_t * plan, int n_own,
                           int64_t K, int64_t F, int64_t N, int k, int64_t T, void * scratch, float * out, int64_t col_lo);
int    q4x_experts_wide_cols();
bool   q4x_experts_wide(void * stream, int ty_gu, const void * up, const void * gate, int64_t row_bytes_gu,
                        int64_t expert_bytes_gu, int ty_down, const void * down, int64_t row_bytes_down,
                        int64_t expert_bytes_down, const float * x, int64_t x_stride, const int32_t * plan, int n_own,
                        int64_t K, int64_t F, int64_t N, int k, int64_t T, void * scratch, float * out);
void   q4x_experts_compare(void * stream, const float * mine, const float * ref, const int32_t * ids, int64_t ids_stride,
                           int k, int filler0, int64_t N, int64_t T);
// Debug: AI_Q4X_WAIT_STATS=1 arms the wait/ring timing counters (offload.cu); prints at exit
void offload_wait_stats_init();
// End of a window: *base += layers (fresh sequence numbers for the next replay).
void offload_window_end(void * stream, unsigned * base, int layers);
// AI_ATTN_PRO: q/k head norms + multi-section RoPE + f16 K/V cache writes + gate copy in one launch
// (kernels/attn_prologue.cu). fo: the qkv projection [rows, T], row stride fo_row floats; kc/vc rows in halves.
// Token-embedding rows from a Q4_K table (ids i32 [T] -> f32 [ncols, T]), the CPU dequantize_row_q4_K arithmetic
bool embed_q4k(void * stream, const void * table, int64_t ncols, int64_t row_bytes, const int32_t * ids, int64_t T,
               float * out);
bool attn_prologue_ok(int64_t D, int64_t n_dims);
// AI_ATTN_OGATE: out = sigmoid(gate) * attn (ggml's fused unary_gated_op) and its q8_1 copy, [ncols, T]
bool attn_ogate_q8(void * stream, const float * attn, const float * gate, float * out, void * q8, int64_t ncols, int64_t T);
void attn_ogate_compare(void * stream, const float * a, const float * b, int64_t n);   // counted with attn_prologue's
bool attn_prologue(void * stream, const float * fo, int64_t fo_row, int64_t Hq, int64_t Hkv, int64_t T, const float * qw,
                   const float * kw, float eps, const int32_t * pos, int n_dims, const int * sections, bool is_imrope,
                   float freq_base, float freq_scale, const int64_t * cells, void * kc, int64_t kc_row, void * vc,
                   int64_t vc_row, float * q_out, float * gate_out, bool rows_by_token);
// the same with q|gate, k and v from three separate projections (artifacts whose q/k/v types differ)
bool attn_prologue3(void * stream, const float * qg, int64_t qg_row, const float * kp, int64_t k_row, const float * vp,
                    int64_t v_row, int64_t Hq, int64_t Hkv, int64_t T, const float * qw, const float * kw, float eps,
                    const int32_t * pos, int n_dims, const int * sections, bool is_imrope, float freq_base,
                    float freq_scale, const int64_t * cells, void * kc, int64_t kc_row, void * vc, int64_t vc_row,
                    float * q_out, float * gate_out, bool rows_by_token);
void attn_prologue_compare(void * stream, const float * q_a, const float * q_b, int64_t nq, const float * g_a,
                           const float * g_b, int64_t ng, const void * k_a, const void * kc, int64_t kc_row,
                           const void * v_a, const void * vc, int64_t vc_row, const int64_t * cells, int64_t T,
                           int64_t width);
bool topk_native(void * stream, const float * logits, int64_t nrows, int64_t T, int32_t * ids, float * w,
                 float clamp_val);
void topk_native_compare(void * stream, const int32_t * ia, const float * wa, const int32_t * ib, int64_t ib_tok,
                         const float * wb, int64_t wb_tok, int64_t T);
// AI_FA_Q: tiled split-K decode attention over a q8_0 (kv_type 0) or q4_0 (kv_type 1) KV cache, D = 256,
// G = Hq/Hkv <= 8, L <= 8. Same layout contract as fa_decode_q40, plus the round's write cells (absolute; may be
// null) with cell_base/region so cells past each stream's last written cell are skipped. split: fa_decode_q_split.
int     fa_decode_q_split(int64_t n_kv, int64_t Hkv, int64_t B);
int64_t fa_decode_q_workspace(int64_t n_kv, int64_t rows_total, int split, int64_t Hkv, int64_t B);
int64_t fa_decode_q_ws_bound(int64_t Hq, int64_t Hkv, int64_t B);
bool    fa_decode_q_enabled();   // AI_FA_Q, default on for compute capability 8.x
// AI_FA_Q_CHECK=1: after a verify call (B == 1, L > 1), re-run each row alone at L = 1 into scratch [D*Hq] and
// count bitwise mismatches against out; the totals print at exit. ws is reused (stream-ordered).
bool    fa_decode_q_selfcheck(void * stream, int kv_type, const float * q, const void * k, const void * v,
                              const void * mask, const int64_t * cells, const float * out, float * scratch, float * ws,
                              int64_t D, int64_t Hq, int64_t Hkv, int64_t L, int64_t B, int64_t n_kv, int split,
                              int64_t cell_stride, int64_t head_stride, int64_t stream_stride, int64_t mask_row,
                              int64_t mask_stream, int64_t cell_base, int64_t region, float scale);
bool    fa_decode_q(void * stream, int kv_type, const float * q, const void * k, const void * v, const void * mask,
                    const int64_t * cells, float * out, float * ws, int64_t D, int64_t Hq, int64_t Hkv, int64_t L,
                    int64_t B, int64_t n_kv, int split, int64_t cell_stride, int64_t head_stride, int64_t stream_stride,
                    int64_t mask_row, int64_t mask_stream, int64_t cell_base, int64_t region, float scale);
// AI_FA_NATIVE: split-K decode attention over a q4_0 KV cache, D = 256, GQA, L <= 8 tokens per stream.
// q f32 [D, Hq, B*L] stream-major; k/v point at stream 0's first visible cell; out f32 [D, Hq, B*L].
// Strides in bytes for the cache, in elements (f16) for the mask. Returns false for unsupported shapes.
int64_t fa_decode_q40_workspace(int64_t n_kv, int64_t rows_total, int split);
bool    fa_decode_q40(void * stream, const float * q, const void * k, const void * v, const void * mask, float * out,
                      float * ws, int64_t D, int64_t Hq, int64_t Hkv, int64_t L, int64_t B, int64_t n_kv, int split,
                      int64_t cell_stride, int64_t head_stride, int64_t stream_stride, int64_t mask_row,
                      int64_t mask_stream, float scale);

// Fused DeltaNet decode prologue: CONCAT(history, new) + SSM_CONV + SILU + ggml's gdn_prep,
// four launches per layer in one. qkv is [C, L, B] with element strides (qkv_s1, qkv_s2); ba is the
// fused beta|alpha projection [2*Hv, L, B]; conv_state is [K1, C, B] contiguous and is rolled in
// place. Writes conv_out [C, L, B] (the gated-delta-net op reads v from it) and the packed qh, kh
// [Sk, Hk, L*B] and gh, bh [Hv, L*B] that the op expects when its own fuse flags are off.
// update_state requires L == 1: with more columns the block holding the last one would overwrite
// history the other blocks are still reading. Not bit-identical to the chain it replaces.
void gdn_conv_prep(void * stream, const float * qkv, int64_t qkv_s1, int64_t qkv_s2, const float * ba,
                   int64_t ba_s1, int64_t ba_s2, const float * conv_w, const float * dt_bias, const float * ssm_a,
                   float * conv_state, float * conv_out, float * qh, float * kh, float * gh, float * bh, int C,
                   int K, int Sk, int Hk, int Hv, int64_t L, int64_t B, float eps, bool update_state,
                   float * conv_in = nullptr);   // record mode: also write history ++ new columns [K1+L, C, B]

// AI_GDN16: native chunked Gated DeltaNet prefill (gdn_chunk16.cu): 16-token chunks, parallel fp32 prepare + a
// register-resident state recurrence on TF32 tensor cores. Same semantics as ggml's scalar-gate
// GGML_OP_GATED_DELTA_NET with its fuse flags off: q/k already L2-normed [Sk, Hk, L, B], g = log gate, beta =
// sigmoid(beta) at g[b * gb_seq + t * gb_tok + h], v head h at v + b * v_seq + t * v_tok + h * Sv, out [Sv, Hv, L, B],
// state M[value][key] per (head, seq) updated in place. Sk == Sv == 128.
bool   gdn16_enabled();   // AI_GDN16: default on for cc 8.x and 12.x, opt-in elsewhere
bool   gdn16_ok(int64_t Sk, int64_t Sv, int64_t Hk, int64_t Hv, int64_t L);
size_t gdn16_scratch_bytes(int64_t Hv, int64_t L, int64_t B);
bool   gdn16(void * stream, const float * q, const float * k, int64_t qk_tok, int64_t qk_seq, const float * v,
             int64_t v_tok, int64_t v_seq, const float * g, const float * beta, int64_t gb_tok, int64_t gb_seq,
             float * state, float * out, int64_t Hk, int64_t Hv, int64_t L, int64_t B, float scale, void * scratch);
// AI_GEMM_F16_PF: prefill-width y[N, T] = W[N, K] (F16 row-major) x[K, T] (f32), fp16 tensor cores, fp32 accumulate
// (gemm_f16w_pf.cu). x_tok / y_tok: per-token strides in floats. K % 32 == 0, 16-B aligned x and W; false = not taken.
bool gemm_f16w_pf_enabled();   // AI_GEMM_F16_PF, default off (changes router numerics)
bool gemm_f16w_pf(void * stream, const void * W, const float * x, float * y, int64_t K, int64_t N, int64_t T,
                  int64_t x_tok, int64_t y_tok);
// AI_GDN_CONV_PF: GDN prefill short conv + SiLU without the history concat (gdn_conv_pf.cu, built -use_fast_math to
// match ggml's SSM_CONV bit for bit): x rows of the projection (token stride x_tok, seq stride x_seq floats, C
// channels), history [K-1, C, B] read then updated in place, w [K, C], out [C, L, B]. K in 3..5; false = not taken.
bool gdn_conv_pf_enabled();   // AI_GDN_CONV_PF, default on (verified bit-identical)
bool gdn_conv_pf(void * stream, const float * x, int64_t x_tok, int64_t x_seq, float * hist, const float * w, float * out,
                 int64_t C, int64_t K, int64_t L, int64_t B);
// The unfused prefill path's form: raw conv output [C, L, B] (q | k | v), the ba projection [2 Hv, L, B] (beta |
// alpha, token stride ba_tok floats), dt_bias and ssm_a; applies the ggml op's fused prep (L2 norm, gate, sigmoid).
size_t gdn16_prefill_scratch_bytes(int64_t Hk, int64_t Hv, int64_t L, int64_t B);
bool   gdn16_prefill(void * stream, const float * conv_out, int64_t C, const float * ba, int64_t ba_tok,
                     const float * dt_bias, const float * ssm_a, float eps, float * state, float * out, int64_t Hk,
                     int64_t Hv, int64_t L, int64_t B, void * scratch);

// Parallel argmax over a row (ggml's runs a whole 248k-row reduction on one block). Two launches:
// AMAX partials per row, then a small reduce. Bit-exact with ggml's
// kernel, ties included (lowest index wins). pmax/pidx are scratch of
// argmax_partials_per_row() entries per row.
int  argmax_partials_per_row();
void argmax_rows(void * stream, const float * x, int64_t ncols, int64_t rows, float * pmax, int32_t * pidx,
                 int32_t * dst, const int32_t * map = nullptr);   // map: dst = map[argmax] (draft vocab)

// r = mix + res; p = rms_norm(r) * w -- the residual add fused into the norm that always follows
// it. Bit-exact with the two ops it replaces (same block size, accumulation order and reduction).
// q8 (optional, may be null): a block_q8_1 copy of the normed output, so consumers do not each
// launch their own quantise. Requires ncols % 32 == 0 and the caller to size it ncols/32 blocks per
// row. Same arithmetic as ggml's quantize_q8_1.
void add_rmsnorm(void * stream, const float * mix, const float * res, const float * w, float * r, float * p,
                 void * q8, int64_t ncols, int64_t rows, float eps);

// Producer-side q8_1 staging (see src/kernels/q8stage_bridge.cu). Registration is a host call made
// during capture, in graph order; the consuming GEMV's lookup happens in the same pass and bakes the
// staged pointer into the recorded kernel node, so replays do no lookup.
void q8_stage_clear();
// Drop only the entries whose f32 activation overlaps [data, data + nbytes) -- a rolled-back workspace range.
void q8_stage_invalidate(const void * data, size_t nbytes);
void q8_stage_put(const void * src1_f32, const void * q8, int64_t ne10_padded, int64_t ne11, int64_t ne12,
                  int64_t ne13, int64_t s11, int64_t s12, int64_t s13);
bool q8_stage_enabled();
// Look up a staged q8_1 copy by the f32 activation pointer and exact geometry; nullptr on any mismatch.
const void * q8_stage_get(const void * src1_f32, int64_t ne10_padded, int64_t ne11, int64_t ne12, int64_t ne13,
                          int64_t s11, int64_t s12, int64_t s13);
// AI_Q8_STAGE=3: byte-compare our staged copy against the reference quantiser's output.
void q8_stage_verify(void * stream, const float * x, const void * mine, int64_t ncols, int64_t rows);
// AI_Q8_STAGE=4: fill the staged buffer with the reference quantiser (isolates emission vs plumbing).
void q8_stage_fill_reference(void * stream, const float * x, void * dst, int64_t ncols, int64_t rows);

// Device-side speculation (B streams, stream-major rows: row = b*L + t).
// m[b] = accepted drafts (drafts are ids[b*L + t + 1]); colmask[b*L + t] = t <= m[b] ? 0 : -1e30
void mtp_accept(void * stream, const int32_t * g, const int32_t * ids, int64_t L, int64_t B,
                int32_t * m, float * colmask);
// dst[b] = src[b*L + m[b]], row_bytes per row
void select_rows(void * stream, const void * src, int64_t L, int64_t B, int64_t row_bytes, const int32_t * m,
                 void * dst);
// head step inputs at p = base[b] + m[b] + offset: pos4 [4B], cells [B] (region layout), mask f16 [n_kv, B]
void mtp_step_inputs(void * stream, const int32_t * base, const int32_t * m, const int32_t * delta, int64_t B,
                     int32_t offset, int64_t region_cells, int64_t first_cell, int64_t n_kv, int32_t * pos4,
                     int64_t * cells, uint16_t * mask);
// conv history after m[b]+1 columns, into a [K1, C, B] slot block
// AI_FOLD_PREP: gate/beta masked by exp(colmask) [Hv, L*B] + conv_gather, one launch (bit-exact with the chain)
void gdn_fold_prep(void * stream, const float * gh, const float * bh, const float * colmask, float * gtm, float * btm,
                   int64_t Hv, int64_t L, int64_t B, const float * conv_input, int64_t K1, int64_t C,
                   const int32_t * m, float * conv_dst);
// the MTP head's prefill mask on device: row i attends cells 1..pos0+i (fill_head_inputs' rule)
void head_causal_mask(void * stream, uint16_t * mask, int64_t T, int64_t n_kv, int64_t pos0);
// AI_DECODE_MASK_DEV: a decode/verify round's mask on device, from the per-stream base positions already in the
// ingress (device memory, so a captured round replays it with each round's positions): row b*L + j at p = base[b] + j.
// main (head = 0): cells 0..p open (fill_inputs' rule); head (head = 1): cells 1..p open (fill_head_inputs' rule).
void decode_causal_mask(void * stream, uint16_t * mask, const int32_t * base, int64_t L, int64_t B, int64_t n_kv,
                        int head);
void conv_gather(void * stream, const float * conv_input, int64_t K1, int64_t L, int64_t C, int64_t B,
                 const int32_t * m, float * dst);

// Second stream for host->device copies that overlap compute, and events to order streams.
void * copy_stream_create();
bool   copy_h2d_async(void * stream, void * dst, const void * src, size_t n);
bool   memset_async(void * stream, void * dst, size_t n);   // device-side zero fill, ordered on `stream`
void * event_create();
void   event_record(void * event, void * stream);
void   stream_wait_event(void * stream, void * event);   // `stream` waits; the host does not

// L2 prefetch of n expert slots' row spans in three slot banks (slots < 0 skipped).
void l2_prefetch_experts(void * stream, const void * up, const void * gate, const void * down, int64_t span_up,
                         int64_t span_gate, int64_t span_down, const int32_t * slots, int64_t n);

// Prototype (bench only): Q8_0 GEMV with int8 activations.
void quantize_act_q8(void * stream, const float * x, int8_t * q, float * d, int64_t nb);
void gemv_q8_0(void * stream, const uint8_t * w, int64_t rows, int64_t cols, const int8_t * xq,
               const float * dx, float * y, int rows_per_cta);

// Q8_0 GEMV with no activation quantisation (f32 x, weights dequantised in registers).
// Weights must be planar (see q8_repack_planar): int8 codes [rows][cols]
// then fp16 scales [rows][cols/32], the same byte count as GGUF's interleaved Q8_0.
void q8_repack_planar(const uint8_t * gguf, uint8_t * planar, int64_t rows, int64_t cols);   // host
bool gemv_q8_planar_ok(int64_t cols);
bool gemv_q8_planar_split_ok(int64_t cols, int split);   // split-K needs cols % (split * 256) == 0
// ncols = T activation columns (1..8), x/y column-major contiguous. mode 1/2/4 = shared-x kernel,
// rows per warp, T=1 only; mode -1/-2/-4 = split-K over 1/2/4 warps per row, any T in 1..8.
void gemv_q8_planar(void * stream, const uint8_t * w, int64_t rows, int64_t cols, int64_t ncols, const float * x,
                    float * y, int mode);
// The same split-K kernel reading fp16 activations (half the x bytes), fp32 accumulation.
// split = 1/2/4 warps per row. x_f16 is __half [ncols][cols], column-major contiguous.
void gemv_q8_planar_f16x(void * stream, const uint8_t * w, int64_t rows, int64_t cols, int64_t ncols,
                         const void * x_f16, float * y, int split);
// The same fp16-activation kernel reading GGUF's own interleaved Q8_0 (no repack), so a fused
// span of Q8_0 tensors works as one matrix and prefill keeps using the unchanged bytes.
void gemv_q8_gguf_f16x(void * stream, const uint8_t * w, int64_t rows, int64_t cols, int64_t ncols,
                       const void * x_f16, float * y, int split);
void cast_f32_f16(void * stream, const float * x, void * y, int64_t n);   // n % 4 == 0

// DFlash2 (kernels/dflash2.cu). Round inputs from the verify's accept count m (null m = no verify: anchor g[0]
// at base): context rows p0..p0+block-1 -> cells pos % C; block [anchor, MASK...] at P0.. -> cells C..C+block-1;
// mask f16 [n_cells, block] by the sliding-window rule over the cyclic slots.
void df2_round_inputs(void * stream, const int32_t * base, const int32_t * m, const int32_t * g, int32_t mask_id,
                      int block, int64_t window, int64_t C, int64_t n_cells, int causal, int32_t * ctx_pos,
                      int64_t * ctx_cells, int32_t * blk_ids, int32_t * blk_pos, int64_t * blk_cells, uint16_t * mask,
                      int64_t n_full = 0, int64_t * ctx_lin = nullptr, int64_t * blk_lin = nullptr,
                      uint16_t * mask_full = nullptr);
void copy_i32(void * stream, const int32_t * src, int32_t * dst, int64_t n);
// Greedy coherent selector over top-k candidates (ids/logits [k, nd]), hidden projection hh [rank, nd] and BF16
// codebooks [rank, V]; the first predecessor is anchor_src[anchor_idx]. drafts i32 [nd].
bool df2_select(void * stream, const int32_t * cand, const float * unary, const float * hh, const void * pred_cb,
                const void * succ_cb, const int32_t * anchor_src, int64_t anchor_idx, int k, int rank, int nd,
                int32_t * drafts);
// Exact top-K per row of x f32 [V, rows] (row_stride in floats): ids i32 [K, rows] and their values, ordered by
// value descending then id ascending. work: topk_rows_workspace(rows, K) bytes of device memory.
int64_t topk_rows_workspace(int64_t rows, int K);
// ids[i] = map[ids[i]] for i < n (a truncated draft head's row index -> token id)
void    map_ids(void * stream, int32_t * ids, int64_t n, const int32_t * map);
bool    topk_rows(void * stream, const float * x, int64_t V, int64_t row_stride, int64_t rows, int K, void * work,
                  int32_t * ids, float * vals);

// QSA block inputs for one stream with cell j = position j (kernels/q4x_qsa.cu): cell_blk [n_kv],
// blk_cells [r*n_blocks], blk_pos [4*n_blocks], bias f32 [n_blocks, T], from pos (pos4 section 0, device).
void    q4x_qsa_inputs(void * stream, const int32_t * pos, int64_t T, int64_t n_kv, int64_t r, int64_t n_blocks,
                       int32_t * cell_blk, int32_t * blk_cells, int32_t * blk_pos, float * bias);
// the QSA pooled cache's dirty blocks of a round (kernels/q4x_qsa.cu): dcells i32 [r * M], drows i64 [M], dpos i32 [4 * M]
void    q4x_qsa_dirty(void * stream, const int32_t * pos, int64_t T, int64_t r, int64_t M, int64_t rows, int32_t * dcells,
                      int64_t * drows, int32_t * dpos);

}  // namespace ai::kern
