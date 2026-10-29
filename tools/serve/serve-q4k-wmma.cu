// SPDX-License-Identifier: Apache-2.0
// Q4_K Tensor Core WMMA Matmul Kernel
//
// Marlin-style: dequant Q4_K in shared memory -> wmma Tensor Core matmul
// No global memory round-trip for dequanted weights.
//
// C[M,N] = A[M,K](FP32) x B[N,K](Q4_K)^T -> C[M,N](FP32)
//
// Tile: TILE_M=16, TILE_N=16, TILE_K=32 (one Q4_K sub-block)
// One warp (32 threads) per output tile.
// K loop: K/32 iterations, each processes one sub-block from each of 16 B rows.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define TILE_K 32  // one Q4_K sub-block

// Q4_K block layout (must match ggml)
struct __attribute__((packed)) Q4K {
    half2    dm;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(Q4K) == 144, "Q4K size");

__device__ __forceinline__ void q4k_get_scale_min(int j, const uint8_t * q, uint8_t & sc, uint8_t & m) {
    if (j < 4) {
        sc = q[j]   & 63;
        m  = q[j+4] & 63;
    } else {
        sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m  = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Dequant 32 elements from a Q4_K sub-block into shared memory FP16
__device__ __forceinline__ void dequant_q4k_subblock(
    half * __restrict__ out,      // shared mem, 32 elements
    const Q4K * __restrict__ blk, // Q4_K block
    int sub_idx,                   // sub-block index (0-7)
    int tid)                       // thread ID (0-31)
{
    float dall = __half2float(__low2half(blk->dm));
    float dmin = __half2float(__high2half(blk->dm));

    uint8_t sc, m;
    q4k_get_scale_min(sub_idx, blk->scales, sc, m);

    float d1 = dall * sc;
    float m1 = dmin * m;

    // Each thread handles 1 element (32 threads for 32 elements)
    if (tid < 32) {
        int qs_idx = sub_idx * 16 + tid / 2;
        uint8_t q = blk->qs[qs_idx];
        int qval = (tid & 1) ? (q >> 4) : (q & 0xF);
        out[tid] = __float2half(d1 * qval - m1);
    }
}

// Main kernel: Q4_K x FP32 -> FP32 with WMMA
// Grid: (N/WMMA_N, M/WMMA_M), one block per output tile
// Block: 32 threads (one warp)
__global__ void q4k_wmma_kernel(
    float * __restrict__ dst,        // [M, N] FP32
    const float * __restrict__ src1, // [M, K] FP32 (activations)
    const void * __restrict__ src0,  // [N, K] Q4_K (weights, flattened)
    int M, int N, int K)
{
    const int tile_n = blockIdx.x * WMMA_N;
    const int tile_m = blockIdx.y * WMMA_M;
    const int tid = threadIdx.x;

    if (tile_n >= N || tile_m >= M) return;

    // Accumulator
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Shared memory for tiles
    __shared__ half shared_A[WMMA_M * WMMA_K];  // 16x16 = 512 bytes
    __shared__ half shared_B[WMMA_N * TILE_K];   // 16x32 = 1024 bytes

    const int n_blocks_per_row = (K + 255) / 256;  // Q4_K blocks per weight row
    const Q4K * weights = (const Q4K *)src0;

    // K loop: process 32 elements per iteration (one Q4_K sub-block)
    for (int k = 0; k < K; k += TILE_K) {
        int k_remaining = min(TILE_K, K - k);

        // Dequant weight tile B[WMMA_N, TILE_K] from Q4_K
        // Each of WMMA_N=16 rows needs TILE_K=32 dequanted elements
        for (int row = 0; row < WMMA_N; row++) {
            int n_idx = tile_n + row;
            if (n_idx >= N) continue;

            // Find Q4_K block and sub-block for this row at position k
            int flat_idx = n_idx * K + k;
            int block_idx = flat_idx / 256;
            int pos_in_block = flat_idx % 256;
            int sub_idx = pos_in_block / 32;

            const Q4K * blk = &weights[block_idx];
            dequant_q4k_subblock(&shared_B[row * TILE_K], blk, sub_idx, tid);
        }

        // Load activation tile A[WMMA_M, TILE_K], converting FP32 -> FP16
        for (int i = tid; i < WMMA_M * TILE_K; i += 32) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int m_idx = tile_m + row;
            int k_idx = k + col;
            if (m_idx < M && k_idx < K) {
                shared_A[i] = __float2half(src1[m_idx * K + k_idx]);
            } else {
                shared_A[i] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // WMMA matmul: process TILE_K in two 16-wide steps
        for (int kk = 0; kk < TILE_K && (k + kk) < K; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

            wmma::load_matrix_sync(a_frag, &shared_A[kk], TILE_K);
            wmma::load_matrix_sync(b_frag, &shared_B[kk], TILE_K);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncthreads();
    }

    // Store result
    wmma::store_matrix_sync(&dst[tile_m * N + tile_n], acc, N, wmma::mem_row_major);
}

// Also handle Q5_0 and Q8_0 with the same approach but simpler dequant
// For Q5_0/Q8_0: blocks are 32 elements, matching TILE_K perfectly

struct __attribute__((packed)) Q50 {
    half d; uint8_t qh[4]; uint8_t qs[16];
};

struct __attribute__((packed)) Q80 {
    half d; int8_t qs[32];
};

__global__ void q5_0_wmma_kernel(
    float * __restrict__ dst, const float * __restrict__ src1,
    const void * __restrict__ src0, int M, int N, int K)
{
    const int tile_n = blockIdx.x * WMMA_N;
    const int tile_m = blockIdx.y * WMMA_M;
    const int tid = threadIdx.x;
    if (tile_n >= N || tile_m >= M) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    __shared__ half shared_A[WMMA_M * TILE_K];
    __shared__ half shared_B[WMMA_N * TILE_K];

    const Q50 * weights = (const Q50 *)src0;
    int blocks_per_row = K / 32;

    for (int k = 0; k < K; k += TILE_K) {
        // Dequant Q5_0 weight tile
        for (int row = 0; row < WMMA_N; row++) {
            int n_idx = tile_n + row;
            if (n_idx >= N) continue;
            int blk_idx = n_idx * blocks_per_row + k / 32;
            const Q50 * blk = &weights[blk_idx];
            float d = __half2float(blk->d);
            uint32_t qh = (uint32_t)blk->qh[0] | ((uint32_t)blk->qh[1]<<8) |
                           ((uint32_t)blk->qh[2]<<16) | ((uint32_t)blk->qh[3]<<24);
            if (tid < 32) {
                uint8_t q4 = blk->qs[tid/2];
                int lo = (tid&1) ? (q4>>4) : (q4&0xF);
                int hi = (qh >> tid) & 1;
                shared_B[row * TILE_K + tid] = __float2half(d * (float)((lo|(hi<<4)) - 16));
            }
        }
        // Load activation
        for (int i = tid; i < WMMA_M * TILE_K; i += 32) {
            int row = i / TILE_K, col = i % TILE_K;
            int m_idx = tile_m + row, k_idx = k + col;
            shared_A[i] = (m_idx < M && k_idx < K) ?
                __float2half(src1[m_idx * K + k_idx]) : __float2half(0.0f);
        }
        __syncthreads();
        for (int kk = 0; kk < TILE_K && (k+kk) < K; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, &shared_A[kk], TILE_K);
            wmma::load_matrix_sync(b_frag, &shared_B[kk], TILE_K);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(&dst[tile_m * N + tile_n], acc, N, wmma::mem_row_major);
}

__global__ void q8_0_wmma_kernel(
    float * __restrict__ dst, const float * __restrict__ src1,
    const void * __restrict__ src0, int M, int N, int K)
{
    const int tile_n = blockIdx.x * WMMA_N;
    const int tile_m = blockIdx.y * WMMA_M;
    const int tid = threadIdx.x;
    if (tile_n >= N || tile_m >= M) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    __shared__ half shared_A[WMMA_M * TILE_K];
    __shared__ half shared_B[WMMA_N * TILE_K];

    const Q80 * weights = (const Q80 *)src0;
    int blocks_per_row = K / 32;

    for (int k = 0; k < K; k += TILE_K) {
        for (int row = 0; row < WMMA_N; row++) {
            int n_idx = tile_n + row;
            if (n_idx >= N) continue;
            int blk_idx = n_idx * blocks_per_row + k / 32;
            const Q80 * blk = &weights[blk_idx];
            float d = __half2float(blk->d);
            if (tid < 32)
                shared_B[row * TILE_K + tid] = __float2half(d * (float)blk->qs[tid]);
        }
        for (int i = tid; i < WMMA_M * TILE_K; i += 32) {
            int row = i / TILE_K, col = i % TILE_K;
            int m_idx = tile_m + row, k_idx = k + col;
            shared_A[i] = (m_idx < M && k_idx < K) ?
                __float2half(src1[m_idx * K + k_idx]) : __float2half(0.0f);
        }
        __syncthreads();
        for (int kk = 0; kk < TILE_K && (k+kk) < K; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, &shared_A[kk], TILE_K);
            wmma::load_matrix_sync(b_frag, &shared_B[kk], TILE_K);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(&dst[tile_m * N + tile_n], acc, N, wmma::mem_row_major);
}

// Dispatch
extern "C" int launch_q4k_wmma_matmul(
    void * dst, const void * src0, const void * src1,
    int M, int N, int K, int src0_type, void * stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block(32);  // one warp

    switch (src0_type) {
        case 12:  // Q4_K
            q4k_wmma_kernel<<<grid, block, 0, stream>>>(
                (float*)dst, (const float*)src1, src0, M, N, K);
            break;
        case 6:   // Q5_0
            q5_0_wmma_kernel<<<grid, block, 0, stream>>>(
                (float*)dst, (const float*)src1, src0, M, N, K);
            break;
        case 8:   // Q8_0
            q8_0_wmma_kernel<<<grid, block, 0, stream>>>(
                (float*)dst, (const float*)src1, src0, M, N, K);
            break;
        default:
            return -1;
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}
