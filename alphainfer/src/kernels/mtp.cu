// SPDX-License-Identifier: Apache-2.0
// Device-side speculation kernels: everything that lets verify, accept, fold and the next
// draft chain run as one captured round with one sync. All take B streams laid out
// stream-major (row = b * L + t).
#include "kernels/kernels.h"

#include <cuda_runtime.h>
#include <cstdint>

#include <cstdio>
#include <cstring>

static constexpr float kMaskNeg = -1e30f;

// m[b] = longest prefix with draft == greedy target; colmask[b*L + t] = t <= m[b] ? 0 : kMaskNeg
__global__ static void mtp_accept_k(const int32_t * g, const int32_t * ids, int32_t * m, float * colmask,
                                    long long L, long long B) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) {
        return;
    }
    const long long base = b * L;
    int32_t acc = 0;
    while (acc < L - 1 && ids[base + acc + 1] == g[base + acc]) {
        ++acc;
    }
    m[b] = acc;
    for (long long t = 0; t < L; ++t) {
        colmask[base + t] = t <= acc ? 0.0f : kMaskNeg;
    }
}

// dst[b] = src[b * L + m[b]]  (row_bytes per row)
// Grid (chunks, B): one thread per 16-byte chunk when the row and both bases are 16-byte aligned, else per
// byte. A single-thread device-side memcpy per stream runs byte by byte and is far slower.
__global__ static void select_rows_k(const unsigned char * src, unsigned char * dst, const int32_t * m,
                                     long long L, long long row_bytes, int vec16) {
    const long long b = blockIdx.y;
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned char * s = src + (b * L + m[b]) * row_bytes;
    unsigned char *       d = dst + b * row_bytes;
    if (vec16) {
        if (i < row_bytes / 16) {
            ((uint4 *) d)[i] = ((const uint4 *) s)[i];
        }
    } else if (i < row_bytes) {
        d[i] = s[i];
    }
}

// One head step per stream at position p = base[b] + m[b] + offset.
// Grid (ceil(n_kv/256), B): the mask row is filled by the whole grid; block 0's thread 0 writes the scalars.
// delta (nullable): per-stream RoPE position minus cell index (<= 0 after an image prompt, whose image tokens
// take fewer RoPE positions than KV cells); cells and the mask stay on the cell index.
__global__ static void mtp_step_inputs_k(const int32_t * base, const int32_t * m, const int32_t * delta, int32_t * pos4,
                                         int64_t * cells, uint16_t * mask, long long B, int offset, long long region,
                                         long long first, long long n_kv) {
    const long long b = blockIdx.y;
    const long long j = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long p = (long long) base[b] + m[b] + offset;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const long long rp = p + (delta ? delta[b] : 0);
        pos4[b]       = (int32_t) rp;
        pos4[B + b]   = (int32_t) rp;
        pos4[2*B + b] = (int32_t) rp;
        pos4[3*B + b] = 0;
        cells[b]      = first + b * region + p;
    }
    if (j < n_kv) {
        mask[b * n_kv + j] = (j >= 1 && j <= p) ? (uint16_t) 0x0000 : (uint16_t) 0xFC00;   // head: cells 1..p
    }
}

// The MTP head's prefill mask, generated in the round instead of uploaded (AI_HEAD_MASK_DEV): row i sits at position
// pos0 + i and attends head cells 1..pos0+i (the same rule as Qwen35Hybrid::fill_head_inputs). Values are exactly 0 or
// -inf, so the result is byte-identical to the host-built mask.
__global__ static void head_causal_mask_k(uint16_t * mask, long long T, long long n_kv, long long pos0) {
    const long long i = blockIdx.y;
    const long long j = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T && j < n_kv) {
        const long long p = pos0 + i;
        mask[i * n_kv + j] = (j >= 1 && j <= p) ? (uint16_t) 0x0000 : (uint16_t) 0xFC00;
    }
}

// conv history after m[b]+1 columns: dst[(b*C + c)*K1 + j] = src[(b*C + c)*(K1+L) + m[b] + 1 + j]
__global__ static void conv_gather_k(const float * src, float * dst, const int32_t * m,
                                     long long K1, long long L, long long C, long long B) {
    const long long bc = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (bc >= B * C) {
        return;
    }
    const long long b   = bc / C;
    const long long off = (long long) m[b] + 1;
    const float * s = src + bc * (K1 + L) + off;
    float *       d = dst + bc * K1;
    for (long long j = 0; j < K1; ++j) {
        d[j] = s[j];
    }
}

// The packed fold's prologue in one launch per layer (AI_FOLD_PREP), replacing exp(colmask), two ggml muls and
// conv_gather. Block 0 masks the recorded gate and beta: x * expf(colmask), exactly the ggml chain's
// arithmetic, and colmask is 0 or -1e30 so the factor is exactly 1 or 0 under any exp. Blocks 1.. are
// conv_gather_k's copy, unchanged.
__global__ static void gdn_fold_prep_k(const float * __restrict__ gh, const float * __restrict__ bh,
                                       const float * __restrict__ colmask, float * __restrict__ gtm,
                                       float * __restrict__ btm, long long Hv, long long LB,
                                       const float * __restrict__ csrc, float * __restrict__ cdst,
                                       const int32_t * __restrict__ m, long long K1, long long L, long long C,
                                       long long B) {
    if (blockIdx.x == 0) {
        for (long long i = threadIdx.x; i < Hv * LB; i += blockDim.x) {
            const float f = expf(colmask[i / Hv]);
            gtm[i]        = gh[i] * f;
            btm[i]        = bh[i] * f;
        }
        return;
    }
    const long long bc = (long long) (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if (bc >= B * C) {
        return;
    }
    const long long b   = bc / C;
    const long long off = (long long) m[b] + 1;
    const float * s = csrc + bc * (K1 + L) + off;
    float *       d = cdst + bc * K1;
    for (long long j = 0; j < K1; ++j) {
        d[j] = s[j];
    }
}

namespace ai::kern {

static unsigned blocks_for(long long n, int threads) {
    return (unsigned) ((n + threads - 1) / threads);
}

void mtp_accept(void * stream, const int32_t * g, const int32_t * ids, int64_t L, int64_t B,
                int32_t * m, float * colmask) {
    mtp_accept_k<<<blocks_for(B, 8), 8, 0, (cudaStream_t) stream>>>(g, ids, m, colmask, (long long) L, (long long) B);
}

void select_rows(void * stream, const void * src, int64_t L, int64_t B, int64_t row_bytes, const int32_t * m,
                 void * dst) {
    if (B <= 0 || row_bytes <= 0) {
        return;
    }
    const int vec16 = row_bytes % 16 == 0 && ((uintptr_t) src % 16) == 0 && ((uintptr_t) dst % 16) == 0;
    const long long n = vec16 ? row_bytes / 16 : row_bytes;
    select_rows_k<<<dim3(blocks_for(n, 256), (unsigned) B), 256, 0, (cudaStream_t) stream>>>(
        (const unsigned char *) src, (unsigned char *) dst, m, (long long) L, (long long) row_bytes, vec16);
}

void mtp_step_inputs(void * stream, const int32_t * base, const int32_t * m, const int32_t * delta, int64_t B,
                     int32_t offset, int64_t region_cells, int64_t first_cell, int64_t n_kv, int32_t * pos4,
                     int64_t * cells, uint16_t * mask) {
    if (B <= 0) {
        return;
    }
    mtp_step_inputs_k<<<dim3(blocks_for(n_kv > 0 ? n_kv : 1, 256), (unsigned) B), 256, 0, (cudaStream_t) stream>>>(
        base, m, delta, pos4, cells, mask, (long long) B, offset, (long long) region_cells, (long long) first_cell,
        (long long) n_kv);
}

// Decode mask on the device (AI_DECODE_MASK_DEV), byte-identical to Qwen35Hybrid::fill_inputs (head = 0) / fill_head_inputs (head = 1).
__global__ static void decode_causal_mask_k(uint16_t * mask, const int32_t * base, long long L, long long n_kv,
                                            int head) {
    const long long r = blockIdx.y;   // row b*L + j
    const long long c = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_kv) {
        return;
    }
    const long long p    = (long long) base[r / L] + r % L;
    const bool      open = head ? (c >= 1 && c <= p) : (c <= p);
    mask[r * n_kv + c] = open ? (uint16_t) 0x0000 : (uint16_t) 0xFC00;
}

void decode_causal_mask(void * stream, uint16_t * mask, const int32_t * base, int64_t L, int64_t B, int64_t n_kv,
                        int head) {
    if (L <= 0 || B <= 0 || n_kv <= 0) {
        return;
    }
    decode_causal_mask_k<<<dim3(blocks_for(n_kv, 256), (unsigned) (L * B)), 256, 0, (cudaStream_t) stream>>>(
        mask, base, (long long) L, (long long) n_kv, head);
}

void head_causal_mask(void * stream, uint16_t * mask, int64_t T, int64_t n_kv, int64_t pos0) {
    if (T <= 0 || n_kv <= 0) {
        return;
    }
    head_causal_mask_k<<<dim3(blocks_for(n_kv, 256), (unsigned) T), 256, 0, (cudaStream_t) stream>>>(
        mask, (long long) T, (long long) n_kv, (long long) pos0);
}

void conv_gather(void * stream, const float * conv_input, int64_t K1, int64_t L, int64_t C, int64_t B,
                 const int32_t * m, float * dst) {
    conv_gather_k<<<blocks_for(B * C, 256), 256, 0, (cudaStream_t) stream>>>(
        conv_input, dst, m, (long long) K1, (long long) L, (long long) C, (long long) B);
}

void gdn_fold_prep(void * stream, const float * gh, const float * bh, const float * colmask, float * gtm, float * btm,
                   int64_t Hv, int64_t L, int64_t B, const float * conv_input, int64_t K1, int64_t C,
                   const int32_t * m, float * conv_dst) {
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[fold-prep] LAUNCHED: gate/beta mask + conv gather in one kernel\n");
    }
    gdn_fold_prep_k<<<1 + blocks_for(B * C, 256), 256, 0, (cudaStream_t) stream>>>(
        gh, bh, colmask, gtm, btm, (long long) Hv, (long long) (L * B), conv_input, conv_dst, m, (long long) K1,
        (long long) L, (long long) C, (long long) B);
}

}  // namespace ai::kern
