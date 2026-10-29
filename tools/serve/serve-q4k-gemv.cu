// SPDX-License-Identifier: Apache-2.0
// Fused Q4_K x FP16 GEMV kernels
//
// W4A16: Weight Q4_K (4-bit quantized), Activation FP16
// Dequantizes Q4_K on the fly in registers and FMAs with the FP16 input.
// No global memory round-trip for dequanted weights.
//
// For small M (1-8 users decode), GEMV is more appropriate than GEMM.
// One block per output row, threads cooperate over K dimension.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

// Q4_K block layout (must match ggml's block_q4_K exactly)
struct __attribute__((packed)) Q4KBlock {
    half2    dm;               // d = __low2half(dm), dmin = __high2half(dm)
    uint8_t  scales[12];       // 6-bit packed scales and mins
    uint8_t  qs[128];          // 4-bit quants (256 values packed)
};
static_assert(sizeof(Q4KBlock) == 144, "Q4K block must be 144 bytes");

// Q5_0 block
struct __attribute__((packed)) Q50Block {
    half     d;
    uint8_t  qh[4];
    uint8_t  qs[16];
};
static_assert(sizeof(Q50Block) == 22, "Q5_0 block must be 22 bytes");

// Q8_0 block
struct __attribute__((packed)) Q80Block {
    half     d;
    int8_t   qs[32];
};
static_assert(sizeof(Q80Block) == 34, "Q8_0 block must be 34 bytes");

// Q4_K GEMV kernel
// Computes: out[batch, n] = sum_k(input[batch, k] * dequant(weight[n, k]))
// One block per (batch, n_chunk). Each block handles CHUNK_N output elements.
// Threads cooperate over K dimension with warp reduction.

#define Q4K_GEMV_CHUNK_N 4   // output elements per block
#define Q4K_GEMV_THREADS 128 // threads per block

__device__ __forceinline__ void q4k_get_scale_min(int j, const uint8_t * q, uint8_t & sc, uint8_t & m) {
    if (j < 4) {
        sc = q[j]   & 63;
        m  = q[j+4] & 63;
    } else {
        sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m  = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Q5_K block (ggml block_q5_K: 256 elems, 176 bytes)
// Identical to Q4_K plus a 32-byte qh array holding the 5th (high) bit per element.
struct __attribute__((packed)) Q5KBlock {
    half2   dm;          // d = __low2half(dm), dmin = __high2half(dm)
    uint8_t scales[12];  // 6-bit packed scales/mins (identical layout to Q4_K)
    uint8_t qh[32];      // 5th (high) bit per element
    uint8_t qs[128];     // low 4 bits (identical layout to Q4_K)
};
static_assert(sizeof(Q5KBlock) == 176, "Q5_K block must be 176 bytes");

// Dequant element j (0..255) of a Q5_K block, matching ggml dequantize_row_q5_K:
// Q4_K low-nibble math + one high bit from bit (j>>5) of qh[j&31].
__device__ __forceinline__ float q5k_dequant_elem(const Q5KBlock * blk, float dall, float dmin, int j) {
    const int sub = j >> 5;                 // 0..7: scale sub-block and qh bit index
    uint8_t sc, m;
    q4k_get_scale_min(sub, blk->scales, sc, m);
    const int qs_idx = ((j >> 6) << 5) + (j & 31);
    const uint8_t q = blk->qs[qs_idx];
    const int lo = (j & 32) ? (q >> 4) : (q & 0xF);
    const int hi = (blk->qh[j & 31] >> sub) & 1;
    const int qval = lo | (hi << 4);
    return dall * (float)sc * (float)qval - dmin * (float)m;
}

__global__ void q4k_gemv_kernel(
    half * __restrict__ out,          // [M, N]
    const half * __restrict__ input,  // [M, K]
    const void * __restrict__ weight, // [N, K] in Q4_K blocks
    int M, int N, int K)
{
    const int batch_idx = blockIdx.y;
    const int n_start = blockIdx.x * Q4K_GEMV_CHUNK_N;
    const int tid = threadIdx.x;

    const int n_blocks_per_row = K / 256;  // Q4_K blocks per weight row
    const half * inp = input + (int64_t)batch_idx * K;

    // Each thread accumulates partial sums for CHUNK_N output elements
    float sums[Q4K_GEMV_CHUNK_N] = {0};

    // Iterate over K in chunks of 256 (one Q4_K block)
    for (int kb = 0; kb < n_blocks_per_row; kb++) {
        int k_base = kb * 256;

        // Each output element needs its own Q4_K block
        for (int nc = 0; nc < Q4K_GEMV_CHUNK_N; nc++) {
            int n_idx = n_start + nc;
            if (n_idx >= N) continue;

            // Load Q4_K block for weight[n_idx, k_base..k_base+255]
            const Q4KBlock * blk = (const Q4KBlock *)weight + (int64_t)n_idx * n_blocks_per_row + kb;
            float dall = __half2float(__low2half(blk->dm));
            float dmin = __half2float(__high2half(blk->dm));

            // Each thread processes a subset of the 256 elements
            for (int j = tid; j < 256; j += Q4K_GEMV_THREADS) {
                // Dequant element j from Q4_K block
                int sub_block = j / 32;      // which sub-block (0-7)

                uint8_t sc, m;
                q4k_get_scale_min(sub_block, blk->scales, sc, m);

                // ggml q4_K nibble layout: per 64-elem group, elems 0-31 = low nibbles
                // of qs[group*32 .. +31], elems 32-63 = high nibbles of the same bytes.
                int qs_idx = ((j >> 6) << 5) + (j & 31);
                uint8_t q = blk->qs[qs_idx];
                int qval = (j & 32) ? (q >> 4) : (q & 0xF);

                float w = dall * sc * qval - dmin * m;
                float a = __half2float(inp[k_base + j]);
                sums[nc] += w * a;
            }
        }
    }

    // Warp reduction
    for (int nc = 0; nc < Q4K_GEMV_CHUNK_N; nc++) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sums[nc] += __shfl_down_sync(0xffffffff, sums[nc], offset);
        }
    }

    // Cross-warp reduction via shared memory
    __shared__ float shared[32][Q4K_GEMV_CHUNK_N];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    for (int nc = 0; nc < Q4K_GEMV_CHUNK_N; nc++) {
        if (lane == 0) shared[warp_id][nc] = sums[nc];
    }
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = Q4K_GEMV_THREADS / warpSize;
        for (int nc = 0; nc < Q4K_GEMV_CHUNK_N; nc++) {
            float val = (lane < n_warps) ? shared[lane][nc] : 0.0f;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (lane == 0) {
                int n_idx = n_start + nc;
                if (n_idx < N) {
                    out[(int64_t)batch_idx * N + n_idx] = __float2half(val);
                }
            }
        }
    }
}

// Q5_0 GEMV kernel

__global__ void q5_0_gemv_kernel(
    half * __restrict__ out,
    const half * __restrict__ input,
    const void * __restrict__ weight,
    int M, int N, int K)
{
    const int batch_idx = blockIdx.y;
    const int n_idx = blockIdx.x;
    if (n_idx >= N) return;
    const int tid = threadIdx.x;

    const int n_blocks_per_row = K / 32;
    const half * inp = input + (int64_t)batch_idx * K;

    float sum = 0.0f;

    for (int kb = 0; kb < n_blocks_per_row; kb++) {
        const Q50Block * blk = (const Q50Block *)weight + (int64_t)n_idx * n_blocks_per_row + kb;
        float d = __half2float(blk->d);
        uint32_t qh = (uint32_t)blk->qh[0] | ((uint32_t)blk->qh[1] << 8) |
                       ((uint32_t)blk->qh[2] << 16) | ((uint32_t)blk->qh[3] << 24);

        for (int j = tid; j < 32; j += blockDim.x) {
            uint8_t q4 = blk->qs[j / 2];
            int lo = (j & 1) ? (q4 >> 4) : (q4 & 0xF);
            int hi = (qh >> j) & 1;
            int val = lo | (hi << 4);
            float w = d * (float)(val - 16);
            float a = __half2float(inp[kb * 32 + j]);
            sum += w * a;
        }
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared[32];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < blockDim.x / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            out[(int64_t)batch_idx * N + n_idx] = __float2half(sum);
        }
    }
}

// Q8_0 GEMV kernel

__global__ void q8_0_gemv_kernel(
    half * __restrict__ out,
    const half * __restrict__ input,
    const void * __restrict__ weight,
    int M, int N, int K)
{
    const int batch_idx = blockIdx.y;
    const int n_idx = blockIdx.x;
    if (n_idx >= N) return;
    const int tid = threadIdx.x;

    const int n_blocks_per_row = K / 32;
    const half * inp = input + (int64_t)batch_idx * K;

    float sum = 0.0f;

    for (int kb = 0; kb < n_blocks_per_row; kb++) {
        const Q80Block * blk = (const Q80Block *)weight + (int64_t)n_idx * n_blocks_per_row + kb;
        float d = __half2float(blk->d);

        for (int j = tid; j < 32; j += blockDim.x) {
            float w = d * (float)blk->qs[j];
            float a = __half2float(inp[kb * 32 + j]);
            sum += w * a;
        }
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared[32];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < blockDim.x / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            out[(int64_t)batch_idx * N + n_idx] = __float2half(sum);
        }
    }
}

// Q6_K block (ggml block_q6_K: 256 elems, 210 bytes)
struct __attribute__((packed)) Q6KBlock {
    uint8_t ql[128];     // quants, lower 4 bits
    uint8_t qh[64];      // quants, upper 2 bits
    int8_t  scales[16];  // 8-bit scales per 16 elements
    half    d;           // super-block scale
};
static_assert(sizeof(Q6KBlock) == 210, "Q6_K block must be 210 bytes");

// Dequant element j (0..255) of a Q6_K block, matching ggml dequantize_row_q6_K.
// Branchless (a switch on the quadrant serializes warps 4-way).
__device__ __forceinline__ float q6k_dequant_elem(const Q6KBlock * blk, float d, int j) {
    const int half_idx = j >> 7;         // 0 or 1 (128-element halves)
    const int r = j & 127;
    const int l = r & 31;                // 0..31
    const int quad = r >> 5;             // 0..3
    const uint8_t * ql = blk->ql + half_idx * 64;
    const uint8_t * qh = blk->qh + half_idx * 32;
    const int8_t  * sc = blk->scales + half_idx * 8;
    // quad 0: ql[l]    & 0xF | qh>>0;  quad 1: ql[l+32] & 0xF | qh>>2
    // quad 2: ql[l]    >> 4  | qh>>4;  quad 3: ql[l+32] >> 4  | qh>>6
    const int lo = (ql[((quad & 1) << 5) + l] >> ((quad >> 1) << 2)) & 0xF;
    const int hi = ((qh[l] >> (quad << 1)) & 3) << 4;
    return d * (float)sc[(l >> 4) + (quad << 1)] * (float)((lo | hi) - 32);
}

// Q6_K GEMV kernel (same skeleton as q8_0)
__global__ void q6k_gemv_kernel(
    half * __restrict__ out,
    const half * __restrict__ input,
    const void * __restrict__ weight,
    int M, int N, int K)
{
    const int batch_idx = blockIdx.y;
    const int n_idx = blockIdx.x;
    if (n_idx >= N) return;
    const int tid = threadIdx.x;

    const int n_blocks_per_row = K / 256;
    const half * inp = input + (int64_t)batch_idx * K;

    float sum = 0.0f;
    for (int kb = 0; kb < n_blocks_per_row; kb++) {
        const Q6KBlock * blk = (const Q6KBlock *)weight + (int64_t)n_idx * n_blocks_per_row + kb;
        float d = __half2float(blk->d);
        for (int j = tid; j < 256; j += blockDim.x) {
            float w = q6k_dequant_elem(blk, d, j);
            float a = __half2float(inp[kb * 256 + j]);
            sum += w * a;
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float shared[32];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane < blockDim.x / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) out[(int64_t)batch_idx * N + n_idx] = __float2half(sum);
    }
}

// Dequant element j (0..255) of a Q4_K block (standalone helper for the embed gather).
__device__ __forceinline__ float q4k_dequant_elem(const Q4KBlock * blk, float dall, float dmin, int j) {
    uint8_t sc, m;
    q4k_get_scale_min(j >> 5, blk->scales, sc, m);
    const int qs_idx = ((j >> 6) << 5) + (j & 31);
    const uint8_t q = blk->qs[qs_idx];
    const int qv = (j & 32) ? (q >> 4) : (q & 0xF);
    return dall * (float)sc * (float)qv - dmin * (float)m;
}

// Quant-resident token-embedding gather.
// out[b,:] = dequant(table[token_ids[b], :]) for K-quant tables (Q4_K/Q5_K/Q6_K).
// Keeps a large embedding in GGUF quant format in VRAM (~half the F16 bytes) instead of
// a dequantized F16 resident copy, which helps a large MoE fit entirely on the GPU.
// n_embd must be a multiple of 256 (one super-block).
__global__ void embed_tokens_quant_kernel(
    half * __restrict__ out, const uint8_t * __restrict__ table,
    const int32_t * __restrict__ token_ids, int n_embd, int qtype)
{
    const int b = blockIdx.x;
    const int tok = token_ids[b];
    half * dst = out + (int64_t)b * n_embd;
    const int nblk = n_embd >> 8;
    if (qtype == 14) {  // Q6_K
        const Q6KBlock * row = (const Q6KBlock *)table + (int64_t)tok * nblk;
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
            const Q6KBlock * blk = row + (i >> 8);
            dst[i] = __float2half(q6k_dequant_elem(blk, __half2float(blk->d), i & 255));
        }
    } else if (qtype == 13) {  // Q5_K
        const Q5KBlock * row = (const Q5KBlock *)table + (int64_t)tok * nblk;
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
            const Q5KBlock * blk = row + (i >> 8);
            dst[i] = __float2half(q5k_dequant_elem(blk, __half2float(__low2half(blk->dm)),
                                                   __half2float(__high2half(blk->dm)), i & 255));
        }
    } else if (qtype == 8) {  // Q8_0 (32-elem blocks)
        const Q80Block * row = (const Q80Block *)table + (int64_t)tok * (n_embd >> 5);
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
            const Q80Block * blk = row + (i >> 5);
            dst[i] = __float2half(__half2float(blk->d) * (float)blk->qs[i & 31]);
        }
    } else {  // Q4_K (qtype == 12)
        const Q4KBlock * row = (const Q4KBlock *)table + (int64_t)tok * nblk;
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
            const Q4KBlock * blk = row + (i >> 8);
            dst[i] = __float2half(q4k_dequant_elem(blk, __half2float(__low2half(blk->dm)),
                                                   __half2float(__high2half(blk->dm)), i & 255));
        }
    }
}

extern "C" void launch_embed_tokens_quant(
    half * out, const void * table, const int32_t * token_ids,
    int n_embd, int batch, int qtype, cudaStream_t stream)
{
    if (batch <= 0) return;
    int threads = (n_embd < 256) ? n_embd : 256;
    threads = ((threads + 31) / 32) * 32;
    embed_tokens_quant_kernel<<<batch, threads, 0, stream>>>(
        out, (const uint8_t *)table, token_ids, n_embd, qtype);
}

// Q5_K GEMV kernel (same skeleton as q6_K)
__global__ void q5k_gemv_kernel(
    half * __restrict__ out,
    const half * __restrict__ input,
    const void * __restrict__ weight,
    int M, int N, int K)
{
    const int batch_idx = blockIdx.y;
    const int n_idx = blockIdx.x;
    if (n_idx >= N) return;
    const int tid = threadIdx.x;

    const int n_blocks_per_row = K / 256;
    const half * inp = input + (int64_t)batch_idx * K;

    float sum = 0.0f;
    for (int kb = 0; kb < n_blocks_per_row; kb++) {
        const Q5KBlock * blk = (const Q5KBlock *)weight + (int64_t)n_idx * n_blocks_per_row + kb;
        const float dall = __half2float(__low2half(blk->dm));
        const float dmin = __half2float(__high2half(blk->dm));
        for (int j = tid; j < 256; j += blockDim.x) {
            float w = q5k_dequant_elem(blk, dall, dmin, j);
            float a = __half2float(inp[kb * 256 + j]);
            sum += w * a;
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float shared[32];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane < blockDim.x / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) out[(int64_t)batch_idx * N + n_idx] = __float2half(sum);
    }
}

// C API: fused Q4_K/Q5_0/Q5_K/Q6_K/Q8_0 matmul

extern "C" void launch_q4k_matmul(
    half * out, const half * input, const void * weight,
    int M, int N, int K, int weight_type, cudaStream_t stream)
{
    if (weight_type == 12) {  // Q4_K
        dim3 grid((N + Q4K_GEMV_CHUNK_N - 1) / Q4K_GEMV_CHUNK_N, M);
        q4k_gemv_kernel<<<grid, Q4K_GEMV_THREADS, 0, stream>>>(out, input, weight, M, N, K);
    } else if (weight_type == 6) {  // Q5_0
        dim3 grid(N, M);
        q5_0_gemv_kernel<<<grid, 128, 0, stream>>>(out, input, weight, M, N, K);
    } else if (weight_type == 8) {  // Q8_0
        dim3 grid(N, M);
        q8_0_gemv_kernel<<<grid, 128, 0, stream>>>(out, input, weight, M, N, K);
    } else if (weight_type == 13) {  // Q5_K
        dim3 grid(N, M);
        q5k_gemv_kernel<<<grid, 128, 0, stream>>>(out, input, weight, M, N, K);
    } else if (weight_type == 14) {  // Q6_K
        dim3 grid(N, M);
        q6k_gemv_kernel<<<grid, 128, 0, stream>>>(out, input, weight, M, N, K);
    } else {
        fprintf(stderr, "[q4k-gemv] unsupported weight type %d\n", weight_type);
    }
}

// MoE quant-resident expert dispatch (mul_mat_id style).
// Expert stacks stay in GGUF quant format in VRAM; these kernels dequantize in-register.
// "Pair" p = (token, k-slot): token = p / k_per_tok, expert = ids[p].
// out rows are per pair (not per token); moe_combine_kernel folds K pairs back per token.
// No host involvement: the whole decode MoE step is device-side (CUDA-graph capturable).

// Warp-per-output-row: each block = 4 warps = 4 output rows of one pair. Lanes stride
// the K range in 4-element steps; reduction is a single warp shuffle (no shared memory,
// no __syncthreads); the per-thread work is small at decode, so cheap reductions matter.
// OT = half (activations) or float (logits). Also serves dense decode GEMVs: pass a
// zeroed ids array + expert_stride_bytes=0 + k_per_tok=1 -> w = wbase for every row.
template <int QTYPE, typename OT>
__global__ void moe_gemv_id_kernel(
    OT * __restrict__ out,              // [P, out_stride] (row n_idx < N written)
    const half * __restrict__ input,    // [B, K] token rows
    const uint8_t * __restrict__ wbase, // stacked expert weights (quant blocks or F16)
    const int32_t * __restrict__ ids,   // [P] expert id per pair
    int64_t expert_stride_bytes,
    int k_per_tok, int N, int K, int out_stride)
{
    const int p = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int n_idx = blockIdx.x * 4 + warp;
    if (n_idx >= N) return;
    const int tid = threadIdx.x & 31;       // lane
    const int nthr = 32;                    // lanes per row

    const uint8_t * w = wbase + (int64_t)ids[p] * expert_stride_bytes;
    const half * inp = input + (int64_t)(p / k_per_tok) * K;

    float sum = 0.0f;

    // All threads map across the full K range in 4-element strides (g = 4*g4): every
    // thread active, 1/4 the loop iterations, half2 input loads. 4 consecutive elements
    // never straddle a quant block or sub-block (4 | 32 | 256). Block-header loads
    // (d/dm/scales) hit L1, since the whole thread block reads the same few bytes.
    const int K4 = K >> 2;
    if (QTYPE == 12) {  // Q4_K
        const Q4KBlock * row = (const Q4KBlock *)w + (int64_t)n_idx * (K >> 8);
        for (int g4 = tid; g4 < K4; g4 += nthr) {
            const int g = g4 << 2;
            const Q4KBlock * blk = row + (g >> 8);
            const int j = g & 255;
            float dall = __half2float(__low2half(blk->dm));
            float dmin = __half2float(__high2half(blk->dm));
            uint8_t sc, m;
            q4k_get_scale_min(j >> 5, blk->scales, sc, m);
            const float ds = dall * sc, dm = dmin * m;
            // ggml q4_K nibble layout: per 64-elem group, elems 0-31 = low nibbles of
            // qs[g32..g32+31], elems 32-63 = high nibbles of the same bytes.
            const uint8_t * qb = blk->qs + ((j >> 6) << 5) + (j & 31);
            const int hi = j & 32;
            const int v0 = hi ? (qb[0] >> 4) : (qb[0] & 0xF);
            const int v1 = hi ? (qb[1] >> 4) : (qb[1] & 0xF);
            const int v2 = hi ? (qb[2] >> 4) : (qb[2] & 0xF);
            const int v3 = hi ? (qb[3] >> 4) : (qb[3] & 0xF);
            const half2 i0 = *(const half2 *)(inp + g), i1 = *(const half2 *)(inp + g + 2);
            sum += (ds * v0 - dm) * __half2float(__low2half(i0))
                 + (ds * v1 - dm) * __half2float(__high2half(i0))
                 + (ds * v2 - dm) * __half2float(__low2half(i1))
                 + (ds * v3 - dm) * __half2float(__high2half(i1));
        }
    } else if (QTYPE == 14) {  // Q6_K
        const Q6KBlock * row = (const Q6KBlock *)w + (int64_t)n_idx * (K >> 8);
        for (int g4 = tid; g4 < K4; g4 += nthr) {
            const int g = g4 << 2;
            const Q6KBlock * blk = row + (g >> 8);
            const float d = __half2float(blk->d);
            const half2 i0 = *(const half2 *)(inp + g), i1 = *(const half2 *)(inp + g + 2);
            sum += q6k_dequant_elem(blk, d, (g & 255) + 0) * __half2float(__low2half(i0))
                 + q6k_dequant_elem(blk, d, (g & 255) + 1) * __half2float(__high2half(i0))
                 + q6k_dequant_elem(blk, d, (g & 255) + 2) * __half2float(__low2half(i1))
                 + q6k_dequant_elem(blk, d, (g & 255) + 3) * __half2float(__high2half(i1));
        }
    } else if (QTYPE == 8) {  // Q8_0
        const Q80Block * row = (const Q80Block *)w + (int64_t)n_idx * (K >> 5);
        for (int g4 = tid; g4 < K4; g4 += nthr) {
            const int g = g4 << 2;
            const Q80Block * blk = row + (g >> 5);
            const float d = __half2float(blk->d);
            const int8_t * q = blk->qs + (g & 31);
            const half2 i0 = *(const half2 *)(inp + g), i1 = *(const half2 *)(inp + g + 2);
            sum += d * ((float)q[0] * __half2float(__low2half(i0))
                      + (float)q[1] * __half2float(__high2half(i0))
                      + (float)q[2] * __half2float(__low2half(i1))
                      + (float)q[3] * __half2float(__high2half(i1)));
        }
    } else if (QTYPE == 13) {  // Q5_K
        const Q5KBlock * row = (const Q5KBlock *)w + (int64_t)n_idx * (K >> 8);
        for (int g4 = tid; g4 < K4; g4 += nthr) {
            const int g = g4 << 2;
            const Q5KBlock * blk = row + (g >> 8);
            const float dall = __half2float(__low2half(blk->dm));
            const float dmin = __half2float(__high2half(blk->dm));
            const int j = g & 255;
            const half2 i0 = *(const half2 *)(inp + g), i1 = *(const half2 *)(inp + g + 2);
            sum += q5k_dequant_elem(blk, dall, dmin, j + 0) * __half2float(__low2half(i0))
                 + q5k_dequant_elem(blk, dall, dmin, j + 1) * __half2float(__high2half(i0))
                 + q5k_dequant_elem(blk, dall, dmin, j + 2) * __half2float(__low2half(i1))
                 + q5k_dequant_elem(blk, dall, dmin, j + 3) * __half2float(__high2half(i1));
        }
    } else {  // F16
        const half * row = (const half *)w + (int64_t)n_idx * K;
        for (int g4 = tid; g4 < K4; g4 += nthr) {
            const int g = g4 << 2;
            const half2 w0 = *(const half2 *)(row + g), w1 = *(const half2 *)(row + g + 2);
            const half2 i0 = *(const half2 *)(inp + g), i1 = *(const half2 *)(inp + g + 2);
            float2 p0 = __half22float2(__hmul2(w0, i0));
            float2 p1 = __half22float2(__hmul2(w1, i1));
            sum += p0.x + p0.y + p1.x + p1.y;
        }
    }

    // Warp-only reduction (each warp owns its row)
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (tid == 0) out[(int64_t)p * out_stride + n_idx] = (OT)sum;
}

// out_stride variant: writes out[p, n] with an arbitrary row stride + base offset, so
// separate q/k/v GEMVs (different quant types) can fill the fused QKV buffer slices.
extern "C" void launch_moe_gemv_id_strided(
    half * out, int out_stride, const half * input, const void * wbase, const int32_t * ids,
    int n_pairs, int k_per_tok, int64_t expert_stride_bytes,
    int N, int K, int qtype, cudaStream_t stream)
{
    if (n_pairs <= 0) return;
    dim3 grid((N + 3) / 4, n_pairs);  // 4 warps per block = 4 output rows
    const uint8_t * wb = (const uint8_t *)wbase;
    switch (qtype) {
        case 12: moe_gemv_id_kernel<12, half><<<grid, 128, 0, stream>>>(out, input, wb, ids, expert_stride_bytes, k_per_tok, N, K, out_stride); break;
        case 13: moe_gemv_id_kernel<13, half><<<grid, 128, 0, stream>>>(out, input, wb, ids, expert_stride_bytes, k_per_tok, N, K, out_stride); break;
        case 14: moe_gemv_id_kernel<14, half><<<grid, 128, 0, stream>>>(out, input, wb, ids, expert_stride_bytes, k_per_tok, N, K, out_stride); break;
        case  8: moe_gemv_id_kernel< 8, half><<<grid, 128, 0, stream>>>(out, input, wb, ids, expert_stride_bytes, k_per_tok, N, K, out_stride); break;
        case  1: moe_gemv_id_kernel< 1, half><<<grid, 128, 0, stream>>>(out, input, wb, ids, expert_stride_bytes, k_per_tok, N, K, out_stride); break;
        default: fprintf(stderr, "[moe-gemv-id] unsupported weight type %d\n", qtype);
    }
}

extern "C" void launch_moe_gemv_id(
    half * out, const half * input, const void * wbase, const int32_t * ids,
    int n_pairs, int k_per_tok, int64_t expert_stride_bytes,
    int N, int K, int qtype, cudaStream_t stream)
{
    launch_moe_gemv_id_strided(out, N, input, wbase, ids, n_pairs, k_per_tok,
                               expert_stride_bytes, N, K, qtype, stream);
}

// F32-output variant (logits GEMV feeding batch_argmax)
extern "C" void launch_moe_gemv_id_f32(
    float * out, const half * input, const void * wbase, const int32_t * ids,
    int n_pairs, int k_per_tok, int64_t expert_stride_bytes,
    int N, int K, int qtype, cudaStream_t stream)
{
    if (n_pairs <= 0) return;
    dim3 grid((N + 3) / 4, n_pairs);
    const uint8_t * wb = (const uint8_t *)wbase;
    switch (qtype) {
        case 12: moe_gemv_id_kernel<12, float><<<grid, 128, 0, stream>>>(out, input, wb, ids, expert_stride_bytes, k_per_tok, N, K, N); break;
        case 13: moe_gemv_id_kernel<13, float><<<grid, 128, 0, stream>>>(out, input, wb, ids, expert_stride_bytes, k_per_tok, N, K, N); break;
        case 14: moe_gemv_id_kernel<14, float><<<grid, 128, 0, stream>>>(out, input, wb, ids, expert_stride_bytes, k_per_tok, N, K, N); break;
        case  8: moe_gemv_id_kernel< 8, float><<<grid, 128, 0, stream>>>(out, input, wb, ids, expert_stride_bytes, k_per_tok, N, K, N); break;
        case  1: moe_gemv_id_kernel< 1, float><<<grid, 128, 0, stream>>>(out, input, wb, ids, expert_stride_bytes, k_per_tok, N, K, N); break;
        default: fprintf(stderr, "[moe-gemv-id-f32] unsupported weight type %d\n", qtype);
    }
}

// MMVQ-style dp4a path (quantized activations).
// Activations are quantized per 32-element group to int8 (q8_1-style: d = max|x|/127,
// s = d*sum(q) kept for the q4_K min term), then the GEMV dot runs on integer dp4a
// instead of per-element dequant-to-float chains, as in ggml MMVQ. Not bit-identical
// to the float path because of activation rounding; the error is of the same order as
// ggml's own MMVQ arithmetic.

struct __align__(8) Q81Act {
    float  d;        // scale
    float  s;        // d * sum(q)  (for asymmetric-quant min terms)
    int8_t qs[32];
};
static_assert(sizeof(Q81Act) == 40, "Q81Act must be 40 bytes");

// Quantize [rows, K] F16 activations -> [rows, K/32] Q81Act blocks. Warp per block.
__global__ void moe_q8_act_kernel(Q81Act * __restrict__ out, const half * __restrict__ in, int K) {
    const int row = blockIdx.y;
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (blk * 32 >= K) return;
    const half * x = in + (int64_t)row * K + blk * 32;
    float v = __half2float(x[lane]);
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
    const float d = a / 127.0f;
    const float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    int q = __float2int_rn(v * id);
    q = max(-127, min(127, q));
    int s = q;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    Q81Act * ob = out + (int64_t)row * (K >> 5) + blk;
    ob->qs[lane] = (int8_t)q;
    if (lane == 0) { ob->d = d; ob->s = d * (float)s; }
}

extern "C" void launch_moe_q8_act(void * out, const half * in, int rows, int K, cudaStream_t stream) {
    if (rows <= 0) return;
    dim3 grid((K / 32 + 3) / 4, rows);
    moe_q8_act_kernel<<<grid, 128, 0, stream>>>((Q81Act *)out, in, K);
}

// 2-byte-aligned uint32 load (q6_K blocks are 210 bytes, so only 2-byte aligned)
__device__ __forceinline__ uint32_t ld_u32_2a(const uint8_t * p) {
    const uint16_t * s = (const uint16_t *)p;
    return (uint32_t)s[0] | ((uint32_t)s[1] << 16);
}

// dp4a id-GEMV: warp per output row, lane per 32-element sub-block.
// q4_K: sum(w*a) over a sub-block = d*sc*d_a*sum(q*qa) - dmin*m*s_a  (8 dp4a / 32 elems)
// q6_K: per-16 scales -> 2 partial dots + 2 ones-dots (sum(qa) via dp4a with 0x01010101)
template <int QTYPE, typename OT>
__global__ void moe_gemv_id_q8_kernel(
    OT * __restrict__ out,
    const Q81Act * __restrict__ act,    // [B or P rows, K/32]
    const uint8_t * __restrict__ wbase,
    const int32_t * __restrict__ ids,
    int64_t expert_stride_bytes,
    int k_per_tok, int N, int K, int out_stride)
{
    const int p = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int n_idx = blockIdx.x * 4 + warp;
    if (n_idx >= N) return;
    const int lane = threadIdx.x & 31;

    const uint8_t * w = wbase + (int64_t)ids[p] * expert_stride_bytes;
    const Q81Act * arow = act + (int64_t)(p / k_per_tok) * (K >> 5);

    float sum = 0.0f;

    if (QTYPE == 12) {  // Q4_K
        const int nsb = K >> 5;
        const Q4KBlock * row = (const Q4KBlock *)w + (int64_t)n_idx * (K >> 8);
        for (int sb = lane; sb < nsb; sb += 32) {
            const Q4KBlock * blk = row + (sb >> 3);
            const int j = sb & 7;                    // sub-block within superblock
            float dall = __half2float(__low2half(blk->dm));
            float dmin = __half2float(__high2half(blk->dm));
            uint8_t sc, m;
            q4k_get_scale_min(j, blk->scales, sc, m);
            const uint32_t * qp = (const uint32_t *)(blk->qs + ((j >> 1) << 5));  // qs is 16B into the 144B block -> 4-aligned
            const int shift = (j & 1) << 2;          // even sub-block: lo nibbles; odd: hi
            const Q81Act * ab = arow + sb;
            const int * aq = (const int *)ab->qs;
            int sumi = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint32_t qw = (qp[i] >> shift) & 0x0F0F0F0Fu;
                sumi = __dp4a((int)qw, aq[i], sumi);
            }
            sum += dall * (float)sc * ab->d * (float)sumi - dmin * (float)m * ab->s;
        }
    } else {  // QTYPE == 14, Q6_K
        const int nrun = K >> 5;
        const Q6KBlock * row = (const Q6KBlock *)w + (int64_t)n_idx * (K >> 8);
        for (int run = lane; run < nrun; run += 32) {
            const Q6KBlock * blk = row + (run >> 3);
            const int r8 = run & 7;
            const int half_i = r8 >> 2;
            const int quad = r8 & 3;
            const uint8_t * ql = blk->ql + half_i * 64 + ((quad & 1) << 5);
            const uint8_t * qh = blk->qh + half_i * 32;
            const int8_t * scp = blk->scales + half_i * 8 + (quad << 1);
            const int lsh = (quad >> 1) << 2;
            const int hsh = quad << 1;
            const float d6 = __half2float(blk->d);
            const Q81Act * ab = arow + run;
            const int * aq = (const int *)ab->qs;
            int dot0 = 0, dot1 = 0, one0 = 0, one1 = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint32_t lo = (ld_u32_2a(ql + 4 * i) >> lsh) & 0x0F0F0F0Fu;
                uint32_t hi = ((ld_u32_2a(qh + 4 * i) >> hsh) & 0x03030303u) << 4;
                int qb = (int)(lo | hi);
                if (i < 4) { dot0 = __dp4a(qb, aq[i], dot0); one0 = __dp4a(0x01010101, aq[i], one0); }
                else       { dot1 = __dp4a(qb, aq[i], dot1); one1 = __dp4a(0x01010101, aq[i], one1); }
            }
            float sumi = (float)scp[0] * (float)(dot0 - 32 * one0)
                       + (float)scp[1] * (float)(dot1 - 32 * one1);
            sum += d6 * ab->d * sumi;
        }
    }

    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) out[(int64_t)p * out_stride + n_idx] = (OT)sum;
}

extern "C" void launch_moe_gemv_id_q8(
    half * out, int out_stride, const void * act, const void * wbase, const int32_t * ids,
    int n_pairs, int k_per_tok, int64_t expert_stride_bytes,
    int N, int K, int qtype, cudaStream_t stream)
{
    if (n_pairs <= 0) return;
    dim3 grid((N + 3) / 4, n_pairs);
    const uint8_t * wb = (const uint8_t *)wbase;
    const Q81Act * a = (const Q81Act *)act;
    switch (qtype) {
        case 12: moe_gemv_id_q8_kernel<12, half><<<grid, 128, 0, stream>>>(out, a, wb, ids, expert_stride_bytes, k_per_tok, N, K, out_stride); break;
        case 14: moe_gemv_id_q8_kernel<14, half><<<grid, 128, 0, stream>>>(out, a, wb, ids, expert_stride_bytes, k_per_tok, N, K, out_stride); break;
        default: fprintf(stderr, "[moe-gemv-id-q8] unsupported weight type %d\n", qtype);
    }
}

extern "C" void launch_moe_gemv_id_q8_f32(
    float * out, const void * act, const void * wbase, const int32_t * ids,
    int n_pairs, int k_per_tok, int64_t expert_stride_bytes,
    int N, int K, int qtype, cudaStream_t stream)
{
    if (n_pairs <= 0) return;
    dim3 grid((N + 3) / 4, n_pairs);
    const uint8_t * wb = (const uint8_t *)wbase;
    const Q81Act * a = (const Q81Act *)act;
    switch (qtype) {
        case 12: moe_gemv_id_q8_kernel<12, float><<<grid, 128, 0, stream>>>(out, a, wb, ids, expert_stride_bytes, k_per_tok, N, K, N); break;
        case 14: moe_gemv_id_q8_kernel<14, float><<<grid, 128, 0, stream>>>(out, a, wb, ids, expert_stride_bytes, k_per_tok, N, K, N); break;
        default: fprintf(stderr, "[moe-gemv-id-q8-f32] unsupported weight type %d\n", qtype);
    }
}

// Device-side router: softmax over all experts + top-K select.
// One block per token. Softmax phases are parallel; the K selection passes run on
// thread 0 (E <= ~1K, K <= 16, so trivial). Weights = softmax probs of the selected
// experts, not renormalized (ggml build_moe_ffn norm_w=false).
// Ties break to the smaller expert index (deterministic).
__global__ void moe_topk_kernel(
    int32_t * __restrict__ ids,   // [B, K_used]
    float * __restrict__ wout,    // [B, K_used]
    const half * __restrict__ logits,  // [B, n_expert]
    int n_expert, int k_used)
{
    extern __shared__ float sp[];  // [n_expert] probs
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const half * lrow = logits + (int64_t)b * n_expert;

    __shared__ float red[32];
    const int lane = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int n_warps = blockDim.x / warpSize;

    // 1. load + block max
    float lmax = -1e30f;
    for (int e = tid; e < n_expert; e += blockDim.x) {
        float v = __half2float(lrow[e]);
        sp[e] = v;
        lmax = fmaxf(lmax, v);
    }
    for (int o = warpSize / 2; o > 0; o >>= 1) lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, o));
    if (lane == 0) red[warp_id] = lmax;
    __syncthreads();
    if (tid == 0) {
        float m = red[0];
        for (int w2 = 1; w2 < n_warps; w2++) m = fmaxf(m, red[w2]);
        red[0] = m;
    }
    __syncthreads();
    const float mx = red[0];
    __syncthreads();

    // 2. exp + block sum
    float lsum = 0.0f;
    for (int e = tid; e < n_expert; e += blockDim.x) {
        float v = expf(sp[e] - mx);
        sp[e] = v;
        lsum += v;
    }
    for (int o = warpSize / 2; o > 0; o >>= 1) lsum += __shfl_xor_sync(0xffffffff, lsum, o);
    if (lane == 0) red[warp_id] = lsum;
    __syncthreads();

    // 3. top-K selection + weights (thread 0)
    if (tid == 0) {
        float sum = 0.0f;
        for (int w2 = 0; w2 < n_warps; w2++) sum += red[w2];
        const float inv = 1.0f / sum;
        for (int k = 0; k < k_used; k++) {
            float best = -1.0f;
            int bi = 0;
            for (int e = 0; e < n_expert; e++) {
                if (sp[e] > best) { best = sp[e]; bi = e; }
            }
            ids[(int64_t)b * k_used + k]  = bi;
            float p = best * inv;
            wout[(int64_t)b * k_used + k] = p;
            sp[bi] = -1.0f;
        }
    }
}

extern "C" void launch_moe_topk(
    int32_t * ids, float * wout, const half * logits,
    int n_expert, int k_used, int batch, cudaStream_t stream)
{
    if (batch <= 0) return;
    int threads = 128;
    size_t smem = (size_t)n_expert * sizeof(float);
    moe_topk_kernel<<<batch, threads, smem, stream>>>(ids, wout, logits, n_expert, k_used);
}

// Combine per-pair down outputs into the per-token FP32 accumulator.
// acc[b,:] = sum_k w[b*K+k] * dn[b*K+k,:]   (writes, no pre-zero needed)
__global__ void moe_combine_kernel(
    float * __restrict__ acc,      // [B, n_embd]
    const half * __restrict__ dn,  // [B*K, n_embd]
    const float * __restrict__ w,  // [B*K]
    int k_used, int n_embd)
{
    const int b = blockIdx.x;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float s = 0.0f;
        for (int k = 0; k < k_used; k++) {
            const int64_t p = (int64_t)b * k_used + k;
            s += w[p] * __half2float(dn[p * n_embd + i]);
        }
        acc[(int64_t)b * n_embd + i] = s;
    }
}

extern "C" void launch_moe_combine(
    float * acc, const half * dn, const float * w,
    int k_used, int n_embd, int batch, cudaStream_t stream)
{
    if (batch <= 0) return;
    int threads = (n_embd < 256) ? n_embd : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    moe_combine_kernel<<<batch, threads, 0, stream>>>(acc, dn, w, k_used, n_embd);
}
