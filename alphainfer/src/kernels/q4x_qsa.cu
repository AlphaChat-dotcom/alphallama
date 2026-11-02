// SPDX-License-Identifier: Apache-2.0
// The QSA (qwen4exp indexer sparse attention) block inputs built on the device, for one stream whose
// cell j holds position j and whose used cells are a prefix -- alphallama's ALPHA_QSA_FAST rule
// (llama-memory-hybrid-idx.cpp set_input_qsa), which is the layout Qwen4ExpRun keeps. Built from the round's positions
// (pos4 section 0, device memory), so a captured round replays it with each round's positions.
#include "kernels/kernels.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

static unsigned q4x_blocks_for(long long n, int threads) {
    return (unsigned) ((n + threads - 1) / threads);
}

// n_used = pos[T-1] + 1 (this round's cells are written); n_full = n_used / r full blocks.
//   cell_blk[j]       = j < n_full*r ? j / r : dead            (dead = n_full < n_blocks ? n_full : n_blocks - 1)
//   blk_cells[b*r+i]  = b < n_full ? b*r + i : 0
//   blk_pos[s*nb + b] = b < n_full ? b*r : 0                    (all 4 rope sections)
//   bias[t*nb + b]    = b >= n_full ? -inf : (b*r >= (pos[t]+1)/r*r ? 1e9 : 0);  bias[t*nb + dead] = 1e9 when dead exists
__global__ static void q4x_qsa_inputs_k(const int32_t * pos, long long T, long long n_kv, long long r, long long nb,
                                        int32_t * cell_blk, int32_t * blk_cells, int32_t * blk_pos, float * bias) {
    const long long n_used = (long long) pos[T - 1] + 1;
    const long long n_full = n_used / r;
    const bool      dead   = n_full < nb;
    const long long dead_b = dead ? n_full : nb - 1;
    const long long i      = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long t      = blockIdx.y;
    if (t == 0) {
        if (i < n_kv) {
            cell_blk[i] = (int32_t) (i < n_full * r ? i / r : dead_b);
        }
        if (i < nb * r) {
            const long long b = i / r;
            blk_cells[i] = (int32_t) (b < n_full ? i : 0);
        }
        if (i < nb) {
            const int32_t v = (int32_t) (i < n_full ? i * r : 0);
            for (int s = 0; s < 4; ++s) {
                blk_pos[s * nb + i] = v;
            }
        }
    }
    if (i < nb && t < T) {
        const long long tail = ((long long) pos[t] + 1) / r * r;
        float v = i >= n_full ? -INFINITY : (i * r >= tail ? 1e9f : 0.0f);
        if (dead && i == dead_b) {
            v = 1e9f;
        }
        bias[t * nb + i] = v;
    }
}

// QSA pooled cache: the round's dirty blocks: b0 = pos[0] / r (the first block a write of this round touches) up to the
// last complete block n_full - 1, at most M of them (M = T / r + 2 covers any round). Entry j: block b0 + j when it is
// complete, else the dump row (rows - 1) with member cell 0 -- a fixed shape, so a captured graph stays valid.
__global__ static void q4x_qsa_dirty_k(const int32_t * pos, long long T, long long r, long long M, long long rows,
                                       int32_t * dcells, int64_t * drows, int32_t * dpos) {
    const long long n_full = ((long long) pos[T - 1] + 1) / r;
    const long long b0     = (long long) pos[0] / r;
    const long long i      = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * r) {
        const long long b = b0 + i / r;
        dcells[i] = (int32_t) (b < n_full ? b * r + i % r : 0);
    }
    if (i < M) {
        const long long b = b0 + i;
        const bool      ok = b < n_full;
        drows[i] = ok ? b : rows - 1;
        for (int s = 0; s < 4; ++s) dpos[s * M + i] = (int32_t) (ok ? b * r : 0);
    }
}

__global__ static void q4x_qsa_iota_k(long long n, long long T, int32_t * out) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * T) out[i] = (int32_t) (i % n);
}

namespace ai::kern {

void q4x_qsa_iota(void * stream, int64_t n, int64_t T, int32_t * out) {
    if (n <= 0 || T <= 0) return;
    q4x_qsa_iota_k<<<q4x_blocks_for(n * T, 256), 256, 0, (cudaStream_t) stream>>>((long long) n, (long long) T, out);
}

void q4x_qsa_dirty(void * stream, const int32_t * pos, int64_t T, int64_t r, int64_t M, int64_t rows, int32_t * dcells,
                   int64_t * drows, int32_t * dpos) {
    if (T <= 0 || M <= 0) return;
    q4x_qsa_dirty_k<<<q4x_blocks_for(M * r, 256), 256, 0, (cudaStream_t) stream>>>(pos, (long long) T, (long long) r,
                                                                                   (long long) M, (long long) rows, dcells,
                                                                                   drows, dpos);
}

void q4x_qsa_inputs(void * stream, const int32_t * pos, int64_t T, int64_t n_kv, int64_t r, int64_t n_blocks,
                    int32_t * cell_blk, int32_t * blk_cells, int32_t * blk_pos, float * bias) {
    if (T <= 0 || n_kv <= 0) {
        return;
    }
    const long long n = n_kv > n_blocks * r ? n_kv : n_blocks * r;
    q4x_qsa_inputs_k<<<dim3(q4x_blocks_for(n, 256), (unsigned) T), 256, 0, (cudaStream_t) stream>>>(
        pos, (long long) T, (long long) n_kv, (long long) r, (long long) n_blocks, cell_blk, blk_cells, blk_pos, bias);
}

}  // namespace ai::kern
