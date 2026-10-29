// SPDX-License-Identifier: Apache-2.0
// Token-embedding gather from a Q4_K table (some artifacts store token_embd.weight as Q4_K; ggml's CUDA
// get_rows has no Q4_K path -- llama.cpp gathers it on the CPU). One block per (token, 256-value super-block),
// one thread per value, the CPU reference's arithmetic (ggml-quants.c dequantize_row_q4_K): d1 = d*sc,
// m1 = dmin*m, y = d1*q - m1, with explicit round-to-nearest ops so no FMA contraction can change a bit.
#include "kernels/kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

struct ek_block_q4_K {
    __half  d, dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(ek_block_q4_K) == 144, "block_q4_K is 144 bytes");

__device__ __forceinline__ void ek_scale_min(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

__global__ void embed_q4k_k(const ek_block_q4_K * __restrict__ table, long long row_blocks,
                            const int32_t * __restrict__ ids, float * __restrict__ out, long long ncols) {
    const long long t   = blockIdx.x;
    const int       sb  = blockIdx.y;
    const int       tid = threadIdx.x;   // 0..255
    const ek_block_q4_K * b = table + (long long) ids[t] * row_blocks + sb;
    const int j    = tid / 64;           // 64-value group
    const int half = (tid % 64) / 32;    // low nibbles first, then high
    const int l    = tid % 32;
    uint8_t sc, m;
    ek_scale_min(2 * j + half, b->scales, sc, m);
    const float d    = __half2float(b->d);
    const float dmin = __half2float(b->dmin);
    const float d1   = __fmul_rn(d, (float) sc);
    const float m1   = __fmul_rn(dmin, (float) m);
    const uint8_t q  = b->qs[32 * j + l];
    const int     nb = half ? (q >> 4) : (q & 0xF);
    out[t * ncols + (long long) sb * 256 + j * 64 + half * 32 + l] = __fsub_rn(__fmul_rn(d1, (float) nb), m1);
}

}   // namespace

namespace ai::kern {

bool embed_q4k(void * stream, const void * table, int64_t ncols, int64_t row_bytes, const int32_t * ids, int64_t T,
               float * out) {
    if (ncols % 256 != 0 || row_bytes != ncols / 256 * 144 || T < 1) {
        return false;
    }
    embed_q4k_k<<<dim3((unsigned) T, (unsigned) (ncols / 256)), 256, 0, (cudaStream_t) stream>>>(
        (const ek_block_q4_K *) table, (long long) (ncols / 256), ids, out, (long long) ncols);
    return true;
}

}   // namespace ai::kern
