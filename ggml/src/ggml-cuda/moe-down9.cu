// Fused MoE down projection (ALPHA_MOE_DOWN9), see moe-down9.cuh
#include "moe-down9.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <cstdio>
#include <cstdlib>

#define MOE_DOWN9_MAX_PATHS 16
#define MOE_DOWN9_MAX_ADD    4

// host-side mirror of the device type table below (a __device__ constexpr cannot be called from host code)
static bool d9_type_supported(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
}

typedef float (*d9_vec_dot_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
                              const int & kbx, const int & iqs);

static constexpr __device__ d9_vec_dot_t d9_vec_dot(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q4_K: return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K: return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K: return vec_dot_q6_K_q8_1;
        default:             return nullptr;
    }
}

// vdr per type. get_vdr_mmvq lives inside mmvq.cu (file-local), so mirror the public VDR_* macros
// from vecdotq.cuh here; a mismatch would change the lane->quant-slice mapping and the result.
static constexpr __host__ __device__ int d9_vdr(ggml_type type) {
    return type == GGML_TYPE_Q8_0 ? VDR_Q8_0_Q8_1_MMVQ :
           type == GGML_TYPE_Q4_K ? VDR_Q4_K_Q8_1_MMVQ :
           type == GGML_TYPE_Q5_K ? VDR_Q5_K_Q8_1_MMVQ :
           type == GGML_TYPE_Q6_K ? VDR_Q6_K_Q8_1_MMVQ : 1;
}

// One warp's dot product of a single weight row against one q8_1 activation row. The lane->(block,
// quant slice) mapping is the MMVQ one with nwarps == 1, so each lane calls vec_dot over its own
// slice of K and the warp reduces.
template <ggml_type type>
static __device__ __forceinline__ float d9_row_dot(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        const int row_block_base, const int blocks_per_row) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = d9_vdr(type);
    constexpr d9_vec_dot_t dot = d9_vec_dot(type);

    const int lane = threadIdx.x;
    const int kqs  = vdr * (lane % (qi / vdr));

    float sum = 0.0f;
    for (int kbx = lane / (qi / vdr); kbx < blocks_per_row; kbx += vdr * WARP_SIZE / qi) {
        const int kby = kbx * (qk / QK8_1);
        sum += dot(vx, &y[kby], row_block_base + kbx, kqs);
    }
    return warp_reduce_sum(sum);
}

// One block per output row, one warp per path: warps 0..k-1 take the routed experts under the routed
// codec, and warp k (when present) takes the shared expert under its own codec, so the codec is
// chosen per warp index. Warp 0 merges the paths in fixed rank order and folds in the addends, so
// the result is reproducible run to run.
template <ggml_type type_r, ggml_type type_s, bool has_shared>
static __global__ void k_moe_down9(
        const void * __restrict__ vx_r, const void * __restrict__ vx_s,
        const block_q8_1 * __restrict__ y_r, const block_q8_1 * __restrict__ y_s,
        const int32_t * __restrict__ ids, const float * __restrict__ w,
        const float * __restrict__ score,
        const float * a0, const float * a1, const float * a2, const float * a3,
        float * __restrict__ dst,
        const int n_embd, const int k, const int n_add,
        const int blocks_per_row_r, const int blocks_per_row_s,
        const int64_t s_row_r, const int64_t s_exp_r, const int64_t s_row_s, const int64_t s_sc1,
        const int64_t ids_s1, const int64_t sw1, const int64_t sw2,
        const int64_t y_r_slot, const int64_t y_r_tok, const int64_t y_s_tok,
        const int64_t sa0_0, const int64_t sa1_0, const int64_t sa0_1, const int64_t sa1_1,
        const int64_t sa0_2, const int64_t sa1_2, const int64_t sa0_3, const int64_t sa1_3,
        const int64_t dst_s1) {
    __shared__ float paths[MOE_DOWN9_MAX_PATHS + 1];

    const int row  = blockIdx.x;
    const int t    = blockIdx.y;
    const int warp = threadIdx.y;

    if (warp < k) {
        const int   eid = ids[t * ids_s1 + warp];
        const float wj  = w[t * sw2 + warp * sw1];
        const float d   = d9_row_dot<type_r>(vx_r, y_r + t * y_r_tok + warp * y_r_slot,
                                             (int) (eid * s_exp_r + row * s_row_r), blocks_per_row_r);
        if (threadIdx.x == 0) {
            paths[warp] = wj * d;
        }
    } else if (has_shared && warp == k) {
        // the shared expert has no routing dependency: its own codec, its own activation
        const float d = d9_row_dot<type_s>(vx_s, y_s + t * y_s_tok, (int) (row * s_row_s), blocks_per_row_s);
        // score carries a row stride like every other input. It is [1, T], so only nb[1] matters,
        // but a fused router|score projection passes a view whose row pitch is the full router width,
        // not 1. Indexing score[t] directly would read another expert's logit for every T > 1.
        const float g = 1.0f / (1.0f + expf(-score[t * s_sc1]));
        if (threadIdx.x == 0) {
            paths[k] = g * d;
        }
    }
    __syncthreads();

    if (warp == 0 && threadIdx.x == 0) {
        const int n_path = has_shared ? k + 1 : k;
        float acc = paths[0];
        for (int j = 1; j < n_path; ++j) {
            acc = acc + paths[j];
        }
        if (n_add > 0) { acc += a0[row * sa0_0 + t * sa1_0]; }
        if (n_add > 1) { acc += a1[row * sa0_1 + t * sa1_1]; }
        if (n_add > 2) { acc += a2[row * sa0_2 + t * sa1_2]; }
        if (n_add > 3) { acc += a3[row * sa0_3 + t * sa1_3]; }
        dst[row + t * dst_s1] = acc;
    }
}

bool ggml_cuda_moe_down9_width_ok(int64_t T) {
    // Exact by default: refuse the fused path above the MMVQ batch bound of 8. Above it the
    // reference switches to MMQ and no warp count reproduces that, so "fused == reference" is
    // not available at any tuning. ALPHA_MOE_DOWN9_EXACT=0 lifts the bound.
    static const bool exact_required = [] {
        const char * e = getenv("ALPHA_MOE_DOWN9_EXACT");
        return !(e && e[0] == '0');
    }();
    return !exact_required || T <= 8;
}

bool ggml_cuda_moe_down9_supported(const ggml_tensor * down_exps, const ggml_tensor * act_routed,
                                   const ggml_tensor * ids, const ggml_tensor * weights,
                                   const ggml_tensor * down_shexp, const ggml_tensor * act_shexp,
                                   const ggml_tensor * score,
                                   const ggml_tensor * const * addends, int n_add,
                                   const ggml_tensor * dst) {
    if (!down_exps || !act_routed || !ids || !weights || !dst || n_add < 0 || n_add > MOE_DOWN9_MAX_ADD) {
        return false;
    }
    if (!d9_type_supported(down_exps->type)) {
        return false;
    }
    if (act_routed->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    const int64_t n_ff   = down_exps->ne[0];
    const int64_t n_embd = down_exps->ne[1];
    const int64_t k      = act_routed->ne[1];
    const int64_t T      = act_routed->ne[2];

    // Width ceiling, checked here so every caller of supported() inherits it instead of each
    // one remembering to call width_ok separately.
    if (!ggml_cuda_moe_down9_width_ok(T)) {
        return false;
    }

    for (int j = 0; j < n_add; ++j) {
        if (!addends[j] || addends[j]->type != GGML_TYPE_F32) {
            return false;
        }
        if (addends[j]->ne[0] != n_embd || addends[j]->ne[1] != T) {
            return false;
        }
    }
    if (n_ff % ggml_blck_size(down_exps->type) != 0) {
        return false;
    }
    // one warp per path, so the path count bounds the block size
    if (k < 1 || k > MOE_DOWN9_MAX_PATHS || T < 1 || act_routed->ne[3] != 1) {
        return false;
    }
    if (ids->ne[0] < k || ids->ne[1] != T) {
        return false;
    }
    if (weights->ne[0] != 1 || weights->ne[1] != k || weights->ne[2] != T) {
        return false;
    }
    if (dst->ne[0] != n_embd || dst->ne[1] != T || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (!ggml_is_contiguous(act_routed)) {
        return false;
    }
    // the shared expert is optional; when present it must match the routed geometry
    const bool has_shared = down_shexp || act_shexp || score;
    if (has_shared) {
        if (!down_shexp || !act_shexp || !score) {
            return false;
        }
        if (!d9_type_supported(down_shexp->type) || act_shexp->type != GGML_TYPE_F32 ||
            score->type != GGML_TYPE_F32) {
            return false;
        }
        if (down_shexp->ne[0] != n_ff || down_shexp->ne[1] != n_embd) {
            return false;
        }
        if (act_shexp->ne[0] != n_ff || act_shexp->ne[1] != T || !ggml_is_contiguous(act_shexp)) {
            return false;
        }
        if (score->ne[0] != 1 || score->ne[1] != T) {
            return false;
        }
        // ne[0] == 1 so only the row pitch matters, and it is threaded to the kernel -- but the element
        // stride must still be a plain float or nb[1] does not describe the layout.
        if (score->nb[0] != sizeof(float)) {
            return false;
        }
        if (n_ff % ggml_blck_size(down_shexp->type) != 0) {
            return false;
        }
        if (k + 1 > MOE_DOWN9_MAX_PATHS) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_op_moe_down9(ggml_backend_cuda_context & ctx,
                            const ggml_tensor * down_exps, const ggml_tensor * act_routed,
                            const ggml_tensor * ids, const ggml_tensor * weights,
                            const ggml_tensor * down_shexp, const ggml_tensor * act_shexp,
                            const ggml_tensor * score,
                            const ggml_tensor * const * addends, int n_add,
                            ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_moe_down9_supported(down_exps, act_routed, ids, weights,
                                              down_shexp, act_shexp, score, addends, n_add, dst));
    cudaStream_t stream = ctx.stream();

    const int64_t n_ff   = down_exps->ne[0];
    const int64_t n_embd = down_exps->ne[1];
    const int     k      = (int) act_routed->ne[1];
    const int64_t T      = act_routed->ne[2];
    const bool    shared = down_shexp != nullptr;

    const int64_t ne_padded = GGML_PAD(n_ff, MATRIX_ROW_PADDING);
    const int64_t blk_row   = ne_padded / QK8_1;   // q8_1 blocks per activation row

    // stage the routed activations as q8_1 with the MMVQ layout: [blocks per row][k][T]
    ggml_cuda_pool_alloc<char> q_r(ctx.pool(), k * T * blk_row * sizeof(block_q8_1));
    {
        const int64_t s11 = act_routed->nb[1] / sizeof(float);
        const int64_t s12 = act_routed->nb[2] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) act_routed->data, nullptr, q_r.get(), down_exps->type,
                               n_ff, s11, s12, s12 * T, ne_padded, k, (int) T, 1, stream);
    }
    ggml_cuda_pool_alloc<char> q_s(ctx.pool());
    if (shared) {
        q_s.alloc(T * blk_row * sizeof(block_q8_1));
        const int64_t s11 = act_shexp->nb[1] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) act_shexp->data, nullptr, q_s.get(), down_shexp->type,
                               n_ff, s11, s11 * T, s11 * T, ne_padded, (int) T, 1, 1, stream);
    }

    const int64_t fs = (int64_t) sizeof(float);
    const float * ap[MOE_DOWN9_MAX_ADD] = { nullptr, nullptr, nullptr, nullptr };
    int64_t as0[MOE_DOWN9_MAX_ADD] = { 0, 0, 0, 0 };
    int64_t as1[MOE_DOWN9_MAX_ADD] = { 0, 0, 0, 0 };
    for (int j = 0; j < n_add; ++j) {
        ap[j]  = (const float *) addends[j]->data;
        as0[j] = (int64_t) addends[j]->nb[0] / fs;
        as1[j] = (int64_t) addends[j]->nb[1] / fs;
    }

    const int64_t ts_r = ggml_type_size(down_exps->type);
    const int64_t ts_s = shared ? ggml_type_size(down_shexp->type) : 1;
    const int      nw  = shared ? k + 1 : k;
    const dim3 grid((unsigned) n_embd, (unsigned) T, 1);
    const dim3 blk(WARP_SIZE, (unsigned) nw, 1);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[moe-down9] fused down+merge: routed %s x%d%s%s, %d addend(s), grid %d x %d, %d warps\n",
                ggml_type_name(down_exps->type), k,
                shared ? " + shared " : "", shared ? ggml_type_name(down_shexp->type) : "",
                n_add, (int) n_embd, (int) T, nw);
    }

#define D9_LAUNCH(TR, TS, HS)                                                                               \
    k_moe_down9<TR, TS, HS><<<grid, blk, 0, stream>>>(                                                      \
        down_exps->data, shared ? down_shexp->data : nullptr,                                               \
        (const block_q8_1 *) q_r.get(), shared ? (const block_q8_1 *) q_s.get() : nullptr,                  \
        (const int32_t *) ids->data, (const float *) weights->data,                                         \
        shared ? (const float *) score->data : nullptr,                                                     \
        ap[0], ap[1], ap[2], ap[3], (float *) dst->data,                                                    \
        (int) n_embd, k, n_add,                                                                             \
        (int) (n_ff / ggml_blck_size(TR)), shared ? (int) (n_ff / ggml_blck_size(TS)) : 0,                   \
        (int64_t) (down_exps->nb[1] / ts_r), (int64_t) (down_exps->nb[2] / ts_r),                            \
        shared ? (int64_t) (down_shexp->nb[1] / ts_s) : 0,                                                  \
        shared ? (int64_t) (score->nb[1] / sizeof(float)) : 0,                                              \
        (int64_t) (ids->nb[1] / sizeof(int32_t)),                                                            \
        (int64_t) (weights->nb[1] / sizeof(float)), (int64_t) (weights->nb[2] / sizeof(float)),               \
        blk_row, blk_row * k, blk_row,                                                                        \
        as0[0], as1[0], as0[1], as1[1], as0[2], as1[2], as0[3], as1[3],                                      \
        (int64_t) (dst->nb[1] / sizeof(float)))

#define D9_BY_ROUTED(TS, HS)                                                     \
    switch (down_exps->type) {                                                   \
        case GGML_TYPE_Q5_K: D9_LAUNCH(GGML_TYPE_Q5_K, TS, HS); break;           \
        case GGML_TYPE_Q6_K: D9_LAUNCH(GGML_TYPE_Q6_K, TS, HS); break;           \
        case GGML_TYPE_Q4_K: D9_LAUNCH(GGML_TYPE_Q4_K, TS, HS); break;           \
        case GGML_TYPE_Q8_0: D9_LAUNCH(GGML_TYPE_Q8_0, TS, HS); break;           \
        default: GGML_ABORT("moe-down9: unsupported routed type");               \
    }

    if (!shared) {
        D9_BY_ROUTED(GGML_TYPE_Q8_0, false)
    } else {
        switch (down_shexp->type) {
            case GGML_TYPE_Q8_0: D9_BY_ROUTED(GGML_TYPE_Q8_0, true) break;
            case GGML_TYPE_Q5_K: D9_BY_ROUTED(GGML_TYPE_Q5_K, true) break;
            case GGML_TYPE_Q6_K: D9_BY_ROUTED(GGML_TYPE_Q6_K, true) break;
            case GGML_TYPE_Q4_K: D9_BY_ROUTED(GGML_TYPE_Q4_K, true) break;
            default: GGML_ABORT("moe-down9: unsupported shared type");
        }
    }
#undef D9_BY_ROUTED
#undef D9_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
