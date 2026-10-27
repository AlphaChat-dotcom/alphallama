// Hand-Tuned Small-M GEMM Kernel
//
// Optimized for decode batches where M=1-8 (tall-and-skinny).
// cuBLAS uses generic kernels not tuned for small M.
// This kernel tiles along N and K with shared memory,
// keeping all M rows resident in registers.
//
// C[M,N] = A[M,K] x B[N,K]^T  (B stored as [N,K] row-major)
// In column-major (cuBLAS convention):
//   C[N,M] = B[K,N]^T x A[K,M]  with lda=K, ldb=K, ldc=N
//
// Each thread block computes a [TILE_N, M] output tile.
// Threads tile over K with shared memory for B (weight) reuse.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Tile sizes tuned for sm_89
#define SMALL_M_TILE_N  128    // output columns per block
#define SMALL_M_TILE_K  32     // K-dimension tile for shared memory
#define SMALL_M_MAX_M   16     // max supported M

// Each thread handles one (n, m) output element's K-reduction
// Block: (TILE_N, M) threads = 128 x 8 = 1024 threads
// Each thread accumulates one dot product over K

__global__ void small_m_gemm_kernel(
    half * __restrict__ C,        // [N, M] col-major, ld=N
    const half * __restrict__ A,  // [K, M] col-major, ld=K
    const half * __restrict__ B,  // [K, N] col-major, ld=K  (= B[N,K] row-major transposed)
    int M, int N, int K)
{
    // Block processes output columns [block_n, block_n + TILE_N)
    const int block_n = blockIdx.x * SMALL_M_TILE_N;

    // Thread indices
    const int n_local = threadIdx.x;  // 0..TILE_N-1
    const int m_idx = threadIdx.y;     // 0..M-1
    const int n_global = block_n + n_local;

    if (n_global >= N || m_idx >= M) return;

    // Accumulator
    float acc = 0.0f;

    // Shared memory for A tile [TILE_K, M] and B tile [TILE_K, TILE_N]
    // A is small (M<=8), so it fits easily
    __shared__ half smem_A[SMALL_M_TILE_K][SMALL_M_MAX_M];
    __shared__ half smem_B[SMALL_M_TILE_K][SMALL_M_TILE_N];

    // Tile over K
    for (int k0 = 0; k0 < K; k0 += SMALL_M_TILE_K) {
        // Load A tile: A[k0:k0+TILE_K, 0:M]
        // Collaborative load: each thread loads one element
        int load_idx = threadIdx.y * blockDim.x + threadIdx.x;
        int total_A = SMALL_M_TILE_K * M;
        for (int i = load_idx; i < total_A; i += blockDim.x * blockDim.y) {
            int kk = i / M;
            int mm = i % M;
            int k_global = k0 + kk;
            smem_A[kk][mm] = (k_global < K) ? A[k_global * M + mm] : __float2half(0.0f);
            // A is [K,M] col-major with ld=K: element (row=k, col=m) at offset m * K + k
        }

        // Load B tile: B[k0:k0+TILE_K, block_n:block_n+TILE_N]
        int total_B = SMALL_M_TILE_K * SMALL_M_TILE_N;
        for (int i = load_idx; i < total_B; i += blockDim.x * blockDim.y) {
            int kk = i / SMALL_M_TILE_N;
            int nn = i % SMALL_M_TILE_N;
            int k_global = k0 + kk;
            int n_g = block_n + nn;
            smem_B[kk][nn] = (k_global < K && n_g < N) ? B[n_g * K + k_global] : __float2half(0.0f);
            // B is [K,N] col-major with ld=K: element (k,n) at offset n * K + k
        }

        __syncthreads();

        // Compute partial dot product
        int k_end = min(SMALL_M_TILE_K, K - k0);
        for (int kk = 0; kk < k_end; kk++) {
            acc += __half2float(smem_A[kk][m_idx]) * __half2float(smem_B[kk][n_local]);
        }

        __syncthreads();
    }

    // Write output
    // C is [N,M] col-major with ld=N: element (n,m) at offset m * N + n
    C[m_idx * N + n_global] = __float2half(acc);
}

extern "C" int launch_cutlass_gemm_small_m(
    half * C, const half * A, const half * B,
    int M, int N, int K, cudaStream_t stream)
{
    if (M > SMALL_M_MAX_M || M < 1) return -1;

    // Grid: one block per TILE_N columns
    int n_blocks = (N + SMALL_M_TILE_N - 1) / SMALL_M_TILE_N;

    // Block: (TILE_N, M) threads
    dim3 block(SMALL_M_TILE_N, M);
    dim3 grid(n_blocks);

    small_m_gemm_kernel<<<grid, block, 0, stream>>>(C, A, B, M, N, K);

    return 0;
}
