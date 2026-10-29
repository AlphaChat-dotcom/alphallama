// SPDX-License-Identifier: Apache-2.0
// DFlash2 drafter device helpers: per-round inputs computed from the verify's accept count, and the
// coherent selector. Both exist so the drafter can sit inside the captured decode round with no host trip.
#include "kernels/kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>

namespace ai::kern {

// Context cache: cyclic over C = window + block cells (slot = pos % C), then `block` scratch cells for the
// block's own K/V, then padding up to n_cells. The verify appends all `block` rows at p0..p0+block-1; rows
// past the accepted ones sit at positions >= the next anchor, so the mask hides them and the next round
// overwrites them, and with C = window + block they can never land on a slot still inside the window.
//
// Slot s holds kp(s) = (P0-1) - ((P0-1-s) mod C): the newest position < P0 congruent to s. A block query at
// P = P0+q sees it iff kp >= 0 and P - kp < window (dflash/model.py _attention_mask, non-causal, sliding).
// causal (DFlash v1 sliding layers): a block query q also stops at block cell C + q. ctx_lin/blk_lin and mask_full
// (may be null) serve a full-attention layer (v1's last layer): linear cells, cell = position; the block is written
// at its own positions (over uncommitted verify rows, which are invisible and rewritten next round) and every cell
// below the block's end is visible to every block query (bidirectional, no window), over n_full cells.
__global__ static void df2_round_inputs_k(const int32_t * base, const int32_t * m, const int32_t * g, int32_t mask_id,
                                          int block, long long window, long long C, long long n_cells, int causal,
                                          int32_t * ctx_pos, int64_t * ctx_cells, int32_t * blk_ids,
                                          int32_t * blk_pos, int64_t * blk_cells, uint16_t * mask,
                                          long long n_full, int64_t * ctx_lin, int64_t * blk_lin,
                                          uint16_t * mask_full) {
    const long long p0 = base[0];
    const long long mm = m ? m[0] : -1;           // m == null: no verify (prefill tail), P0 = p0
    const long long P0 = m ? p0 + mm + 1 : p0;
    const long long j  = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (j < block) {
        ctx_pos[j]   = (int32_t) (p0 + j);
        ctx_cells[j] = (p0 + j) % C;
        blk_ids[j]   = j == 0 ? (m ? g[mm] : g[0]) : mask_id;
        blk_pos[j]   = (int32_t) (P0 + j);
        blk_cells[j] = C + j;
        if (ctx_lin) {
            ctx_lin[j] = p0 + j;
            blk_lin[j] = P0 + j;
        }
    }
    if (mask_full && j < n_full) {
        const bool vis = j < P0 + block;
        for (int q = 0; q < block; ++q) {
            mask_full[(long long) q * n_full + j] = vis ? (uint16_t) 0x0000 : (uint16_t) 0xFC00;
        }
    }
    if (j < n_cells) {
        long long kp = -1;
        if (j < C) {
            const long long d = ((P0 - 1 - j) % C + C) % C;
            kp = P0 - 1 - d;
        }
        for (int q = 0; q < block; ++q) {
            const long long P = P0 + q;
            bool vis;
            if (j < C) {
                vis = kp >= 0 && P - kp < window;
            } else {
                vis = j < C + block && (!causal || j - C <= q);
            }
            mask[(long long) q * n_cells + j] = vis ? (uint16_t) 0x0000 : (uint16_t) 0xFC00;
        }
    }
}

void df2_round_inputs(void * stream, const int32_t * base, const int32_t * m, const int32_t * g, int32_t mask_id,
                      int block, int64_t window, int64_t C, int64_t n_cells, int causal, int32_t * ctx_pos,
                      int64_t * ctx_cells, int32_t * blk_ids, int32_t * blk_pos, int64_t * blk_cells, uint16_t * mask,
                      int64_t n_full, int64_t * ctx_lin, int64_t * blk_lin, uint16_t * mask_full) {
    const int threads = 256;
    long long n = n_cells > block ? n_cells : block;
    if (mask_full && n_full > n) n = n_full;
    df2_round_inputs_k<<<(unsigned) ((n + threads - 1) / threads), threads, 0, (cudaStream_t) stream>>>(
        base, m, g, mask_id, block, (long long) window, (long long) C, (long long) n_cells, causal, ctx_pos,
        ctx_cells, blk_ids, blk_pos, blk_cells, mask, (long long) n_full, ctx_lin, blk_lin, mask_full);
}

__global__ static void copy_i32_k(const int32_t * src, int32_t * dst, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
void copy_i32(void * stream, const int32_t * src, int32_t * dst, int64_t n) {
    copy_i32_k<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(src, dst, (long long) n);
}

__device__ static float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

// Coherent selector, greedy (dflash/model.py CandidateSelector.select): one block, `rank` threads.
//   for t: S[c] = unary[c,t] + sum_r pred_cb[pred, r] * hh[r, t] * succ_cb[cand[c,t], r];  pred = cand[argmax S]
// Ties go to the lower token id.
__global__ static void df2_select_k(const int32_t * cand, const float * unary, const float * hh,
                                    const uint16_t * pred_cb, const uint16_t * succ_cb, const int32_t * anchor_src,
                                    long long anchor_idx, int k, int rank, int nd, int32_t * drafts) {
    extern __shared__ float sh[];      // rank partials
    __shared__ float   score[64];
    __shared__ int32_t pred_s;
    const int r = threadIdx.x;
    if (r == 0) {
        pred_s = anchor_src[anchor_idx];
    }
    __syncthreads();
    for (int t = 0; t < nd; ++t) {
        const int32_t pred = pred_s;
        const float   ph   = r < rank ? bf16f(pred_cb[(long long) pred * rank + r]) * hh[(long long) t * rank + r] : 0.0f;
        for (int c = 0; c < k; ++c) {
            const int32_t id = cand[(long long) t * k + c];
            float v = r < rank ? ph * bf16f(succ_cb[(long long) id * rank + r]) : 0.0f;
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, o);
            }
            if ((r & 31) == 0) {
                sh[r >> 5] = v;
            }
            __syncthreads();
            if (r == 0) {
                float s = unary[(long long) t * k + c];
                for (int w = 0; w < (int) (blockDim.x >> 5); ++w) {
                    s += sh[w];
                }
                score[c] = s;
            }
            __syncthreads();
        }
        if (r == 0) {
            int best = 0;
            for (int c = 1; c < k; ++c) {
                const int32_t ic = cand[(long long) t * k + c], ib = cand[(long long) t * k + best];
                if (score[c] > score[best] || (score[c] == score[best] && ic < ib)) {
                    best = c;
                }
            }
            pred_s    = cand[(long long) t * k + best];
            drafts[t] = pred_s;
        }
        __syncthreads();
    }
}

bool df2_select(void * stream, const int32_t * cand, const float * unary, const float * hh, const void * pred_cb,
                const void * succ_cb, const int32_t * anchor_src, int64_t anchor_idx, int k, int rank, int nd,
                int32_t * drafts) {
    if (k > 64 || rank % 32 != 0 || rank > 1024) {
        return false;
    }
    df2_select_k<<<1, rank, (size_t) (rank / 32) * sizeof(float), (cudaStream_t) stream>>>(
        cand, unary, hh, (const uint16_t *) pred_cb, (const uint16_t *) succ_cb, anchor_src, (long long) anchor_idx,
        k, rank, nd, drafts);
    return true;
}

// ---- exact top-k over long rows (the drafter's LM-head logits, V = 248320) ----
// ggml_top_k is a full segmented sort, a noticeable share of a DFlash2 round at 7 rows x 248320. Two passes of
// block argmax instead: stage 1 gives each of `parts` blocks per row its chunk's
// top-K, stage 2 picks the row's top-K from parts*K candidates. Order: value descending, then token id
// ascending, so the selection is deterministic and ties go to the lower id.
static constexpr int TK_PARTS   = 128;
static constexpr int TK_THREADS = 256;

__device__ __forceinline__ static bool tk_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

// block-wide argmax of (v, i) held one per thread; result valid in every thread
__device__ __forceinline__ static void tk_block_argmax(float & v, int & i, float * sv, int * si) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, v, o);
        const int   oi = __shfl_down_sync(0xffffffffu, i, o);
        if (tk_better(ov, oi, v, i)) {
            v = ov;
            i = oi;
        }
    }
    if (lane == 0) {
        sv[warp] = v;
        si[warp] = i;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < nw ? sv[lane] : -INFINITY;
        i = lane < nw ? si[lane] : INT_MAX;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, v, o);
            const int   oi = __shfl_down_sync(0xffffffffu, i, o);
            if (tk_better(ov, oi, v, i)) {
                v = ov;
                i = oi;
            }
        }
        if (lane == 0) {
            sv[0] = v;
            si[0] = i;
        }
    }
    __syncthreads();
    v = sv[0];
    i = si[0];
    __syncthreads();
}

__global__ static void tk_stage1(const float * x, long long V, long long row_stride, int K, float * pv, int * pi) {
    extern __shared__ float chunk[];
    __shared__ float sv[32];
    __shared__ int   si[32];
    const int       row = blockIdx.y, part = blockIdx.x;
    const long long per = (V + TK_PARTS - 1) / TK_PARTS;
    const long long b   = (long long) part * per;
    const int       len = (int) max(0LL, min(V, b + per) - b);
    const float *   xr  = x + (long long) row * row_stride + b;
    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        chunk[j] = xr[j];
    }
    __syncthreads();
    for (int r = 0; r < K; ++r) {
        float v = -INFINITY;
        int   i = INT_MAX;
        for (int j = threadIdx.x; j < len; j += blockDim.x) {
            if (tk_better(chunk[j], j, v, i)) {
                v = chunk[j];
                i = j;
            }
        }
        tk_block_argmax(v, i, sv, si);
        if (threadIdx.x == 0) {
            const long long o = ((long long) row * TK_PARTS + part) * K + r;
            pv[o] = v;
            pi[o] = i == INT_MAX ? INT_MAX : (int) (b + i);
            if (i != INT_MAX) {
                chunk[i] = -INFINITY;
            }
        }
        __syncthreads();
    }
}

__global__ static void tk_stage2(const float * pv, const int * pi, int K, int32_t * ids, float * vals) {
    extern __shared__ float cv[];
    __shared__ float sv[32];
    __shared__ int   si[32];
    int *     ci  = (int *) (cv + TK_PARTS * K);
    const int row = blockIdx.x, n = TK_PARTS * K;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        cv[j] = pv[(long long) row * n + j];
        ci[j] = pi[(long long) row * n + j];
    }
    __syncthreads();
    for (int r = 0; r < K; ++r) {
        float v = -INFINITY;
        int   i = INT_MAX, slot = -1;
        for (int j = threadIdx.x; j < n; j += blockDim.x) {   // compare by (value, global token id)
            if (tk_better(cv[j], ci[j], v, i)) {
                v    = cv[j];
                i    = ci[j];
                slot = j;
            }
        }
        float v2 = v;
        int   i2 = i;
        tk_block_argmax(v2, i2, sv, si);
        if (slot >= 0 && v == v2 && i == i2) {   // exactly one thread owns the winner (token ids are unique)
            ids[(long long) row * K + r]  = i2;
            vals[(long long) row * K + r] = v2;
            cv[slot] = -INFINITY;
            ci[slot] = INT_MAX;
        }
        __syncthreads();
    }
}

__global__ static void map_ids_k(int32_t * ids, long long n, const int32_t * map) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ids[i] = map[ids[i]];
    }
}

void map_ids(void * stream, int32_t * ids, int64_t n, const int32_t * map) {
    map_ids_k<<<(unsigned) ((n + 127) / 128), 128, 0, (cudaStream_t) stream>>>(ids, (long long) n, map);
}

int64_t topk_rows_workspace(int64_t rows, int K) {
    return rows * TK_PARTS * K * (int64_t) (sizeof(float) + sizeof(int));
}

bool topk_rows(void * stream, const float * x, int64_t V, int64_t row_stride, int64_t rows, int K, void * work,
               int32_t * ids, float * vals) {
    if (K < 1 || K > 64 || V < 1 || V > INT_MAX / 2) {
        return false;
    }
    float * pv = (float *) work;
    int *   pi = (int *) (pv + rows * TK_PARTS * K);
    const long long per = (V + TK_PARTS - 1) / TK_PARTS;
    tk_stage1<<<dim3(TK_PARTS, (unsigned) rows), TK_THREADS, (size_t) per * sizeof(float), (cudaStream_t) stream>>>(
        x, (long long) V, (long long) row_stride, K, pv, pi);
    tk_stage2<<<(unsigned) rows, TK_THREADS, (size_t) TK_PARTS * K * (sizeof(float) + sizeof(int)),
                (cudaStream_t) stream>>>(pv, pi, K, ids, vals);
    return true;
}

}  // namespace ai::kern
