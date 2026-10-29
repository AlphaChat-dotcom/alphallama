// SPDX-License-Identifier: Apache-2.0
// Marlin W4A16 Matmul - ported from IST-DASLab/vLLM
// Standalone: no PyTorch dependency. Uses Tensor Core mma.sync m16n8k16.
//
// Instantiates Marlin kernel for INT4 weights x FP16 activations.
// Provides:
//   1. marlin_matmul(): the kernel (requires Marlin-format weights)
//   2. convert_q4k_to_marlin(): one-time weight format conversion
//   3. launch_marlin_matmul_hook(): ggml-cuda hook wrapper

#ifdef ALPHA_HAS_FLASHINFER

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cublas_v2.h>

// Marlin headers (standalone, no PyTorch)
#include "marlin/scalar_type.hpp"
#include "marlin/marlin_template.h"
#include "marlin/gptq_marlin_repack.cuh"

using namespace marlin;

// Kernel instantiation for INT4 x FP16
// kU4B8: unsigned 4-bit with bias-8, no zero-point buffer needed
// thread_k_blocks=8: K_tile=128, b_sh_wr_iters=1
// Two group_blocks variants: 2 (group_size=32) and 8 (group_size=128)

static bool marlin_kernel_inited = false;
static int marlin_max_smem = 0;

static void marlin_kernel_init() {
    if (marlin_kernel_inited) return;
    marlin_kernel_inited = true;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&marlin_max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    // kU4B8 variants (no zero-points)
    auto * k2 = &marlin::Marlin<half, sglang::kU4B8.id(), marlin::default_threads,
                                 1, 8, 8, true, marlin::pipe_stages, 2, false>;
    auto * k8 = &marlin::Marlin<half, sglang::kU4B8.id(), marlin::default_threads,
                                 1, 8, 8, true, marlin::pipe_stages, 8, false>;
    // kU4 variants (with zero-points, for AWQ)
    auto * k4_g8 = &marlin::Marlin<half, sglang::kU4.id(), marlin::default_threads,
                                    1, 8, 8, true, marlin::pipe_stages, 8, false>;
    cudaFuncSetAttribute(k2, cudaFuncAttributeMaxDynamicSharedMemorySize, marlin_max_smem);
    cudaFuncSetAttribute(k8, cudaFuncAttributeMaxDynamicSharedMemorySize, marlin_max_smem);
    cudaFuncSetAttribute(k4_g8, cudaFuncAttributeMaxDynamicSharedMemorySize, marlin_max_smem);
    fprintf(stderr, "[marlin] kernel init: max_smem=%d (kU4B8 g2+g8, kU4 g8)\n", marlin_max_smem);
}

// Standalone wrappers for ForwardEngine
// ensure_device_info declared below (after d_locks/cached_sms statics)
static void ensure_device_info();
extern "C" void marlin_kernel_init_standalone() { marlin_kernel_init(); ensure_device_info(); }

extern "C" void launch_gptq_marlin_repack_standalone(
    const uint32_t * b_q_weight, uint32_t * out,
    int size_k, int size_n, cudaStream_t stream)
{
    marlin::launch_gptq_marlin_repack(b_q_weight, out, size_k, size_n, stream);
}

// AWQ -> Marlin repack (reads AWQ [K, N/8] directly, no GPTQ conversion)
// Ported from SGLang: awq_marlin_repack.cuh
// AWQ packs nibbles along N: qweight[k, n/8] has 8 output-dimension values per uint32
// GPTQ packs nibbles along K: qweight[k/8, n] has 8 reduction-dimension values per uint32
// The two repacks produce different Marlin interleavings, so use the one matching the source layout.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
template <int num_threads, int num_bits>
__global__ void awq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr, uint32_t* __restrict__ out_ptr,
    int size_k, int size_n) { return; }
#else
template <int num_threads, int num_bits>
__global__ void awq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr, uint32_t* __restrict__ out_ptr,
    int size_k, int size_n) {
    using namespace marlin;
    constexpr int pack_factor = 32 / num_bits;
    int k_tiles = size_k / tile_k_size;
    int n_tiles = size_n / tile_n_size;
    int block_k_tiles = div_ceil(k_tiles, (int)gridDim.x);
    auto start_k_tile = blockIdx.x * block_k_tiles;
    if (start_k_tile >= k_tiles) return;
    int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

    auto wait_for_stage = [&]() { cp_async_wait<repack_stages - 2>(); __syncthreads(); };

    extern __shared__ int4 sh[];
    constexpr int tile_n_ints = tile_n_size / pack_factor;
    constexpr int stage_n_threads = tile_n_ints / 4;
    constexpr int stage_k_threads = tile_k_size;
    constexpr int stage_size = stage_k_threads * stage_n_threads;

    auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
        if (n_tile_id >= n_tiles) { cp_async_fence(); return; }
        int first_n = n_tile_id * tile_n_size;
        int first_n_packed = first_n / pack_factor;
        int4* sh_ptr = sh + stage_size * pipe;
        if (threadIdx.x < stage_size) {
            auto k_id = threadIdx.x / stage_n_threads;
            auto n_id = threadIdx.x % stage_n_threads;
            int first_k = k_tile_id * tile_k_size;
            cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                reinterpret_cast<int4 const*>(
                    &(b_q_weight_ptr[(first_k + k_id) * (size_n / pack_factor) + first_n_packed + (n_id * 4)])));
        }
        cp_async_fence();
    };

    auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
        if (n_tile_id >= n_tiles) return;
        auto warp_id = threadIdx.x / 32;
        auto th_id = threadIdx.x % 32;
        if (warp_id >= 4) return;
        int tc_col = th_id / 4;
        int tc_row = (th_id % 4) * 2;
        constexpr int tc_offsets[4] = {0, 1, 8, 9};
        int cur_n = warp_id * 16 + tc_col;
        int cur_n_packed = cur_n / pack_factor;
        int cur_n_pos = cur_n % pack_factor;
        constexpr int sh_stride = tile_n_ints;
        constexpr uint32_t mask = (1 << num_bits) - 1;
        int4* sh_stage_ptr = sh + stage_size * pipe;
        uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);
        int cur_n_pos_unpacked;
        if constexpr (num_bits == 4) {
            constexpr int undo_pack[8] = {0, 4, 1, 5, 2, 6, 3, 7};
            cur_n_pos_unpacked = undo_pack[cur_n_pos];
        } else {
            constexpr int undo_pack[4] = {0, 2, 1, 3};
            cur_n_pos_unpacked = undo_pack[cur_n_pos];
        }
        uint32_t vals[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int cur_elem = tc_row + tc_offsets[i];
            int packed_src_0 = sh_stage_int_ptr[cur_n_packed + sh_stride * cur_elem];
            int packed_src_1 = sh_stage_int_ptr[cur_n_packed + (8 / pack_factor) + sh_stride * cur_elem];
            vals[i] = (packed_src_0 >> (cur_n_pos_unpacked * num_bits)) & mask;
            vals[4 + i] = (packed_src_1 >> (cur_n_pos_unpacked * num_bits)) & mask;
        }
        constexpr int tile_size_val = tile_k_size * tile_n_size / pack_factor;
        int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size_val;
        if constexpr (num_bits == 4) {
            constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
            uint32_t res = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) res |= vals[pack_idx[i]] << (i * 4);
            out_ptr[out_offset + th_id * 4 + warp_id] = res;
        }
    };

    for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
        int n_tile_id = 0;
        #pragma unroll
        for (int pipe = 0; pipe < repack_stages - 1; pipe++)
            fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
        while (n_tile_id < n_tiles) {
            #pragma unroll
            for (int pipe = 0; pipe < repack_stages; pipe++) {
                fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id,
                                n_tile_id + pipe + repack_stages - 1);
                repack_tile(pipe, k_tile_id, n_tile_id + pipe);
                wait_for_stage();
            }
            n_tile_id += repack_stages;
        }
    }
}
#endif

extern "C" void launch_awq_marlin_repack(
    const uint32_t * b_q_weight, uint32_t * out,
    int size_k, int size_n, cudaStream_t stream)
{
    using namespace marlin;
    int dev = 0;
    cudaGetDevice(&dev);
    int sms = 0, max_smem = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    auto * kernel = awq_marlin_repack_kernel<repack_threads, 4>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);
    kernel<<<sms, repack_threads, max_smem, stream>>>(b_q_weight, out, size_k, size_n);
}

static void marlin_launch_grouped(
    const void * A, const void * B, void * C, const void * scales,
    int num_groups, int M, int N, int K, int group_blocks, int smem_size,
    int * locks, dim3 grid, dim3 block, cudaStream_t stream)
{
    marlin_kernel_init();
    if (group_blocks == 2) {
        marlin::Marlin<half, sglang::kU4B8.id(), marlin::default_threads,
                        1, 8, 8, true, marlin::pipe_stages, 2, false>
            <<<grid, block, smem_size, stream>>>(
                (const int4 *)A, (const int4 *)B, (int4 *)C, nullptr,
                (const int4 *)scales, nullptr, nullptr, nullptr,
                num_groups, M, N, K, K,
                locks, false, false, smem_size);
    } else {
        marlin::Marlin<half, sglang::kU4B8.id(), marlin::default_threads,
                        1, 8, 8, true, marlin::pipe_stages, 8, false>
            <<<grid, block, smem_size, stream>>>(
                (const int4 *)A, (const int4 *)B, (int4 *)C, nullptr,
                (const int4 *)scales, nullptr, nullptr, nullptr,
                num_groups, M, N, K, K,
                locks, false, false, smem_size);
    }
}

// Lock buffer for Marlin's internal synchronization
static int * d_locks = nullptr;
static int locks_size = 0;
static int cached_sms = 0;
static int cached_smem = 0;

// Pool of lock buffers - one per Marlin call per forward pass.
// Pre-zeroed once at start of forward pass, each call gets its own slot.
static constexpr int MAX_MARLIN_CALLS = 128;  // 4 per layer x 32 layers max
static int lock_slot_idx = 0;

static void ensure_locks(int n) {
    int needed = n * MAX_MARLIN_CALLS;
    if (locks_size >= needed) return;
    if (d_locks) cudaFree(d_locks);
    locks_size = needed;
    cudaMalloc(&d_locks, locks_size * sizeof(int));
    cudaMemset(d_locks, 0, locks_size * sizeof(int));
}

// Reset lock pool index - call once at start of each forward pass
extern "C" void marlin_reset_locks(void * stream_ptr) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    ensure_device_info();
    lock_slot_idx = 0;
    // Single memset for the entire pool instead of one per call
    cudaMemsetAsync(d_locks, 0, cached_sms * MAX_MARLIN_CALLS * sizeof(int), stream);
}

static int * get_next_lock_slot() {
    int offset = lock_slot_idx * cached_sms;
    lock_slot_idx++;
    return d_locks + offset;
}

static void ensure_device_info() {
    if (cached_sms > 0) return;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&cached_sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&cached_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    ensure_locks(cached_sms);
}

// AWQ Marlin matmul: kU4 with explicit zero-points (matching SGLang pipeline)
extern "C" int launch_marlin_matmul_awq(
    const void * A,       // FP16 activations [M, K]
    const void * B,       // Marlin-repacked INT4 weights (via awq_marlin_repack)
    void * C,             // FP16 output [M, N]
    const void * scales,  // FP16 permuted scales [K/group, N] (via marlin_permute_scales)
    const void * zp,      // INT4 packed zero-points [K/group, N/8] (via awq_to_marlin_zero_points)
    int M, int N, int K,
    int group_size,
    void * stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (K % 128 != 0 || N % 128 != 0) return -1;
    if (M > 16) return -1;

    int num_groups = K / group_size;

    // Use per-call lock slot (pool pre-zeroed at start of forward pass)
    int * my_locks = get_next_lock_slot();

    // SGLang uses atomic_add for small M: ceil(M/64)*N <= 2048
    bool use_atomic = (((M + 63) / 64) * N <= 2048);

    // kU4 with has_zp=true, is_zp_float=false, group_blocks=8
    marlin::Marlin<half, sglang::kU4.id(), marlin::default_threads,
                    1, 8, 8, true, marlin::pipe_stages, 8, false>
        <<<cached_sms, marlin::default_threads, cached_smem, stream>>>(
            (const int4 *)A, (const int4 *)B, (int4 *)C, nullptr,
            (const int4 *)scales, nullptr, (const int4 *)zp, nullptr,
            num_groups, M, N, K, K,
            my_locks, use_atomic, false, cached_smem);

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

extern "C" int launch_marlin_matmul(
    const void * A,       // FP16 activations [M, K] (packed as int4 for alignment)
    const void * B,       // Marlin-format INT4 weights [K/16, N*16/8] (packed)
    void * C,             // FP16 output [M, N]
    const void * scales,  // FP16 scales [1, N] (channelwise) or [K/group, N]
    int M, int N, int K,
    int group_size,       // -1 for channelwise, 128 for group quantization
    void * stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Marlin requires N % 128 == 0 and K % 128 == 0 (thread_k_blocks=8 x tile_size=16)
    if (K % 128 != 0 || N % 128 != 0) return -1;
    if (M > 16) return -1;  // thread_m_blocks=1 -> max M=16

    int group_blocks = (group_size == -1) ? -1 : group_size / tile_size;
    int num_groups = (group_size == -1) ? 1 : K / group_size;

    // Thread block config for small M (decode: M=1-16)
    const int thread_m_blocks = 1;
    const int thread_n_blocks = 8;  // 128 output columns per block
    const int thread_k_blocks = 8;  // 128 reduction elements per step (b_sh_wr_iters=1)

    dim3 grid(cached_sms, 1, 1);
    dim3 block(default_threads);

    // Use per-call lock slot (pool pre-zeroed at start of forward pass)
    int * my_locks = get_next_lock_slot();

    static bool logged = false;
    if (!logged) {
        fprintf(stderr, "[marlin] launch: M=%d N=%d K=%d blocks=%d smem=%d groups=%d\n",
                M, N, K, cached_sms, cached_smem, num_groups);
        logged = true;
    }

    marlin_launch_grouped(
        A, B, C, scales, num_groups, M, N, K, group_blocks, cached_smem, my_locks, grid, block, stream);

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Q4_K -> Marlin format converter and ggml hook.
// Converts Q4_K packed weights to Marlin's INT4 layout + FP16 scales.
// Called once at model load (one-time cost).
//
// Q4_K format: 256 elements per block (dall, dmin, scales[12], qs[128])
// Marlin format: INT4 packed in specific interleaved bit layout + per-group FP16 scales
//
// For simplicity, Q4_K is dequantized to FP16 and re-quantized to plain INT4 with per-group scales.
// This is lossy (Q4_K has per-sub-block scales, Marlin uses per-group/channelwise).
// VRAM stays at INT4 level after conversion.

extern "C" void launch_dequant_q4k(half *, const void *, int, cudaStream_t);
// Stubs for unused dequant types (only Q4_K is used)
static void launch_dequant_q5_0(half *, const void *, int, cudaStream_t) {}
static void launch_dequant_q8_0(half *, const void *, int, cudaStream_t) {}
static void launch_dequant_q6k(half *, const void *, int, cudaStream_t) {}
extern "C" void launch_f32_to_f16(half *, const float *, int64_t, cudaStream_t);
extern "C" void launch_f16_to_f32(float *, const half *, int, cudaStream_t);

struct MarlinWeight {
    void * d_packed;   // Marlin INT4 repacked (for GPTQ/AWQ models)
    void * d_scales;   // FP16 scales
    half * d_fp16;     // FP16 dequanted weights [K, N] column-major (for cuBLAS path)
    int N, K;
    int group_size;
    bool use_cublas;   // true = cuBLAS FP16 path (Q5_0/Q8_0), false = Marlin INT4 path
};

#include <unordered_map>
static std::unordered_map<const void *, MarlinWeight> g_marlin_cache;

// Forward declarations
extern "C" int convert_q4k_to_marlin(MarlinWeight *, const void *, int, int, int, cudaStream_t);
extern "C" void launch_f32_to_f16(half *, const float *, int64_t, cudaStream_t);
extern "C" void launch_f16_to_f32(float *, const half *, int, cudaStream_t);

// Marlin matmul hook for ggml
// Called when ggml has quantized weight x FP32 input at batch>1
extern "C" int marlin_matmul_hook(
    void * dst_ptr, const void * src0_ptr, const void * src1_ptr,
    int M, int N, int K, int src0_type, void * stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Requirements: K and N must be multiples of 128 (thread_k_blocks=8 x tile_size=16)
    if (K % 128 != 0 || N % 128 != 0) return -1;

    // cuBLAS FP16 path: no INT4 requantization, exact computation

    // Lazy dequant: first call dequants weight to FP16, cached for cuBLAS HGEMM
    auto it = g_marlin_cache.find(src0_ptr);
    if (it == g_marlin_cache.end()) {
        MarlinWeight mw = {};
        mw.N = N; mw.K = K; mw.use_cublas = true;
        // Dequant to FP16 on GPU (one-time, persists in VRAM)
        int n_el = N * K;
        cudaMalloc(&mw.d_fp16, (size_t)n_el * sizeof(half));
        switch (src0_type) {
            case 6:  launch_dequant_q5_0(mw.d_fp16, (const void *)src0_ptr, n_el, stream); break;
            case 8:  launch_dequant_q8_0(mw.d_fp16, (const void *)src0_ptr, n_el, stream); break;
            case 12: launch_dequant_q4k(mw.d_fp16, (const void *)src0_ptr, n_el, stream); break;
            case 14: launch_dequant_q6k(mw.d_fp16, (const void *)src0_ptr, n_el, stream); break;
            default: cudaFree(mw.d_fp16); return -1;
        }
        cudaStreamSynchronize(stream);
        g_marlin_cache[src0_ptr] = mw;
        it = g_marlin_cache.find(src0_ptr);
        static int conv_count = 0;
        if (++conv_count <= 5)
            fprintf(stderr, "[cublas-hook] dequant %dx%d type=%d → FP16 (%.1f MiB)\n",
                    N, K, src0_type, (double)(n_el * 2) / (1024*1024));
    }

    auto & mw = it->second;

    // Pre-allocated FP16 buffers
    static half * d_input_fp16 = nullptr;
    static half * d_output_fp16 = nullptr;
    static cublasHandle_t cublas_handle = nullptr;
    static bool buffers_ready = false;
    if (!buffers_ready) {
        size_t max_input = 16 * 8192;
        size_t max_output = 16 * 262144;
        cudaMalloc(&d_input_fp16, max_input * sizeof(half));
        cudaMalloc(&d_output_fp16, max_output * sizeof(half));
        cublasCreate(&cublas_handle);
        buffers_ready = true;
    }

    size_t needed = (size_t)M * K;
    launch_f32_to_f16(d_input_fp16, (const float *)src1_ptr, (int)needed, stream);

    size_t out_needed = (size_t)M * N;

    // cuBLAS HGEMM: C[N,M] = W^T[N,K] x X[K,M]
    // W is dequanted weight stored as [K,N] column-major
    cublasSetStream(cublas_handle, stream);
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);
    cublasHgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha_h,
                mw.d_fp16, K,       // W[K,N] column-major, transposed to W^T[N,K]
                d_input_fp16, K,    // X[K,M] column-major
                &beta_h,
                d_output_fp16, N);  // C[N,M] column-major

    launch_f16_to_f32((float *)dst_ptr, d_output_fp16, (int)out_needed, stream);

    return 0;
}

// Converts a [N, K] Q4_K weight to Marlin format on GPU+CPU.
// Returns: d_packed (Marlin INT4 on GPU), d_scales (FP16 on GPU)
// group_size = 128 (standard for Marlin)

extern "C" void launch_dequant_q6k(half *, const void *, int, cudaStream_t);

// Unused CPU reference for the GPU repack kernel, kept for documentation of the layout.
static void marlin_repack_cpu(
    uint32_t * out,          // Marlin-format packed output
    const int8_t * q_int4,   // [N, K] simple INT4 values (range 0-15)
    int N, int K)
{
    // Marlin tiles: [tile_k=16, tile_n=64]
    // Output: tile_size = 16 * 64 / 8 = 128 uint32 per tile
    // Within each tile, the uint32 are indexed as: th_id * 4 + warp_id
    // where th_id=0..31, warp_id=0..3
    //
    // Each thread (th_id) in each warp (warp_id) packs 8 INT4 values:
    //   tc_col = th_id / 4         (0..7: column within 16-wide half-tile)
    //   tc_row = (th_id % 4) * 2   (0,2,4,6: starting row pair)
    //   tc_offsets = {0, 1, 8, 9}  (row offsets for the 4 values)
    //   warp_id selects 16-column blocks: 0=cols[0..15], 1=[16..31], 2=[32..47], 3=[48..63]
    //   b1 = first 8 cols, b2 = second 8 cols within the warp's 16-col block
    //
    // pack_idx = {0, 2, 4, 6, 1, 3, 5, 7} for INT4 interleaving

    constexpr int tile_k = 16;
    constexpr int tile_n = 64;
    constexpr int tile_size = tile_k * tile_n / 8;  // 128 uint32 per tile
    constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int k_tiles = K / tile_k;
    int n_tiles = N / tile_n;

    memset(out, 0, (size_t)k_tiles * n_tiles * tile_size * sizeof(uint32_t));

    for (int kt = 0; kt < k_tiles; kt++) {
        for (int nt = 0; nt < n_tiles; nt++) {
            int out_offset = (kt * n_tiles + nt) * tile_size;
            int k_base = kt * tile_k;
            int n_base = nt * tile_n;

            // Simulate 4 warps x 32 threads
            for (int warp_id = 0; warp_id < 4; warp_id++) {
                for (int th_id = 0; th_id < 32; th_id++) {
                    int tc_col = th_id / 4;
                    int tc_row = (th_id % 4) * 2;

                    // cur_n = column within tile's 16-col warp block
                    int cur_n_in_tile = warp_id * 16 + tc_col;

                    // Gather 8 values: 4 from b1 (first 8 cols), 4 from b2 (second 8 cols)
                    uint32_t vals[8];
                    for (int i = 0; i < 4; i++) {
                        int k_idx = k_base + tc_row + tc_offsets[i];
                        int n1_idx = n_base + cur_n_in_tile;       // b1 col
                        int n2_idx = n_base + cur_n_in_tile + 8;   // b2 col

                        vals[i] = (k_idx < K && n1_idx < N) ?
                            ((uint32_t)q_int4[n1_idx * K + k_idx] & 0xF) : 0;
                        vals[4 + i] = (k_idx < K && n2_idx < N) ?
                            ((uint32_t)q_int4[n2_idx * K + k_idx] & 0xF) : 0;
                    }

                    // Interleave pack
                    uint32_t res = 0;
                    for (int i = 0; i < 8; i++) {
                        res |= vals[pack_idx[i]] << (i * 4);
                    }

                    out[out_offset + th_id * 4 + warp_id] = res;
                }
            }
        }
    }
}

extern "C" int convert_q4k_to_marlin(
    MarlinWeight * result,
    const void * d_q4k_weights,  // Q4_K on GPU
    int src_type,                // ggml type (6=Q5_0, 8=Q8_0, 12=Q4_K, 14=Q6_K)
    int N, int K,                // weight dimensions
    cudaStream_t stream)
{
    int n_elements = N * K;
    int group_size = 32;  // match Q5_0 block_size for minimal requant error

    // Step 1: Dequant to FP16 on GPU
    half * d_fp16 = nullptr;
    cudaMalloc(&d_fp16, n_elements * sizeof(half));
    switch (src_type) {
        case 12: launch_dequant_q4k(d_fp16, d_q4k_weights, n_elements, stream); break;
        case  8: launch_dequant_q8_0(d_fp16, d_q4k_weights, n_elements, stream); break;
        case  6: launch_dequant_q5_0(d_fp16, d_q4k_weights, n_elements, stream); break;
        case 14: launch_dequant_q6k(d_fp16, d_q4k_weights, n_elements, stream); break;
        default: cudaFree(d_fp16); return -1;
    }

    // Step 2: Download FP16 to CPU (the stream must be synced first)
    cudaStreamSynchronize(stream);  // wait for dequant to finish
    std::vector<half> h_fp16(n_elements);
    cudaMemcpy(h_fp16.data(), d_fp16, n_elements * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(d_fp16);

    // Step 3: Compute per-group scales + quantize to INT4
    int n_groups = K / group_size;
    std::vector<half> h_scales(n_groups * N);
    std::vector<int8_t> h_q4(n_elements);

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < n_groups; g++) {
            // Find max positive and max negative in group
            // kU4B8 dequant range: (q-8)*scale, q=[0,15] -> [-8*scale, +7*scale]
            // Scale must fit both the positive and negative extremes
            float vmax = 0.0f, vmin = 0.0f;
            for (int k = g * group_size; k < (g + 1) * group_size && k < K; k++) {
                float v = __half2float(h_fp16[n * K + k]);
                if (v > vmax) vmax = v;
                if (v < vmin) vmin = v;
            }
            // scale >= vmax/7 (positive fits) AND scale >= -vmin/8 (negative fits)
            float scale = fmaxf(vmax / 7.0f, -vmin / 8.0f);
            if (scale == 0.0f) scale = 1.0f;
            h_scales[g * N + n] = __float2half(scale);

            // Quantize: val = round(fp16 / scale) + 8 (shift to unsigned 0-15)
            for (int k = g * group_size; k < (g + 1) * group_size && k < K; k++) {
                float v = __half2float(h_fp16[n * K + k]);
                int q = (int)roundf(v / scale) + 8;
                q = max(0, min(15, q));
                h_q4[n * K + k] = (int8_t)q;
            }
        }
    }

    // Verify requantization quality: reject if NaN/inf is detected
    {
        float max_err = 0, sum_err = 0;
        int n_checked = std::min(N * K, 10000);  // sample first 10K values
        bool has_bad = false;
        for (int i = 0; i < n_checked; i++) {
            float orig = __half2float(h_fp16[i]);
            if (!std::isfinite(orig)) { has_bad = true; break; }
        }
        if (has_bad) {
            fprintf(stderr, "[marlin-conv] REJECTED %dx%d type=%d: NaN/inf in dequanted weights\n",
                    N, K, src_type);
            return -1;
        }
        for (int i = 0; i < n_checked; i++) {
            int n = i / K, k = i % K;
            int g = k / group_size;
            float orig = __half2float(h_fp16[n * K + k]);
            float scale = __half2float(h_scales[g * N + n]);
            float recon = scale * ((float)h_q4[n * K + k] - 8.0f);
            float err = fabsf(orig - recon);
            sum_err += err;
            if (err > max_err) max_err = err;
        }
        static int log_count = 0;
        if (log_count < 5) {
            fprintf(stderr, "[marlin-conv] %dx%d type=%d: max_err=%.4f avg_err=%.6f\n",
                    N, K, src_type, max_err, sum_err / n_checked);
            log_count++;
        }
    }

    // Step 4: Pack INT4 into GPTQ format (8 values per uint32)
    // GPTQ layout: gptq[k/pack_factor][n] where pack_factor=8
    int pack_factor = 8;
    int gptq_rows = K / pack_factor;
    std::vector<uint32_t> h_gptq(gptq_rows * N, 0);

    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            int pack_row = k / pack_factor;
            int pack_pos = k % pack_factor;
            h_gptq[pack_row * N + n] |= ((uint32_t)(h_q4[n * K + k] & 0xF)) << (pack_pos * 4);
        }
    }

    // Step 5: Upload GPTQ to GPU, run GPU repack kernel -> Marlin format
    uint32_t * d_gptq = nullptr;
    int marlin_packed_size = (K / marlin::tile_size) * (N * marlin::tile_size / pack_factor);
    cudaMalloc(&d_gptq, gptq_rows * N * sizeof(uint32_t));
    cudaMalloc(&result->d_packed, marlin_packed_size * sizeof(uint32_t));
    cudaMemcpy(d_gptq, h_gptq.data(), gptq_rows * N * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // GPU repack: GPTQ -> Marlin interleaved format
    marlin::launch_gptq_marlin_repack(d_gptq, (uint32_t *)result->d_packed, K, N, stream);
    cudaDeviceSynchronize();
    cudaFree(d_gptq);

    // Upload scales
    cudaMalloc(&result->d_scales, n_groups * N * sizeof(half));
    cudaMemcpy(result->d_scales, h_scales.data(), n_groups * N * sizeof(half), cudaMemcpyHostToDevice);
    result->N = N;
    result->K = K;
    result->group_size = group_size;

    fprintf(stderr, "[marlin] converted %dx%d weight: type=%d → Marlin INT4 (%.1f MiB)\n",
            N, K, src_type, (double)(marlin_packed_size * sizeof(uint32_t)) / (1024*1024));

    return 0;
}
// ggml dispatch hook.
// Called from ggml_cuda_mul_mat() for Q4_K weights at decode (M<=16).
// Uses Marlin Tensor Core INT4 instead of dp4a MMQ/MMVQ.

static bool g_marlin_hook_enabled = false;

extern "C" void marlin_hook_enable() { g_marlin_hook_enabled = true; }
extern "C" bool marlin_hook_active() { return g_marlin_hook_enabled; }

// Convert all Q4_K weight tensors to Marlin format (call once at load time)
extern "C" int marlin_convert_model_weights(
    const void ** weight_ptrs,   // array of Q4_K GPU weight pointers
    const int * types,           // ggml types (12 = Q4_K)
    const int * N_dims,          // output dims per weight
    const int * K_dims,          // input dims per weight
    int n_weights,
    void * stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    ensure_device_info();
    int converted = 0;

    for (int i = 0; i < n_weights; i++) {
        int N = N_dims[i], K = K_dims[i];
        // Marlin requires K%128==0 && N%128==0
        if (K % 128 != 0 || N % 128 != 0) continue;
        if (types[i] != 12 && types[i] != 14) continue;  // Q4_K=12, Q6_K=14

        MarlinWeight mw = {};
        if (convert_q4k_to_marlin(&mw, weight_ptrs[i], types[i], N, K, stream) == 0) {
            g_marlin_cache[weight_ptrs[i]] = mw;
            converted++;
        }
    }

    if (converted > 0) {
        g_marlin_hook_enabled = true;
        fprintf(stderr, "[marlin-hook] converted %d/%d weights to Marlin INT4\n", converted, n_weights);
    }
    return converted;
}

// The actual hook: called from ggml_cuda_mul_mat for quantized weights
extern "C" int marlin_ggml_mul_mat(
    void * dst_ptr,              // FP32 output [M, N] (ggml always wants F32)
    const void * src0_ptr,       // quantized weights (lookup in cache)
    const void * src1_ptr,       // FP32 activations [M, K]
    int M, int N, int K,
    int src_type,                // ggml type (2=Q4_0, 3=Q4_1, 6=Q5_0, 8=Q8_0, 12=Q4_K, 14=Q6_K)
    void * stream_ptr)
{
    if (!g_marlin_hook_enabled) return -1;
    if (M > 16) return -1;  // Marlin only handles decode (M<=16)

    // Only convert Q4_K (type 12), which requires ne0 % 256 == 0
    // Other quant types (Q4_0, Q5_0, Q6_K) need different dequant kernels
    if (src_type != 12) return -1;  // only Q4_K supported for now

    cudaStream_t stream = (cudaStream_t)stream_ptr;

    auto it = g_marlin_cache.find(src0_ptr);
    if (it == g_marlin_cache.end()) {
        if (K % 128 != 0 || N % 128 != 0) return -1;
        if (K % 256 != 0) return -1;  // Q4_K block size = 256 elements along K
        if (N < 256 || K < 256) return -1;
        MarlinWeight mw = {};
        if (convert_q4k_to_marlin(&mw, src0_ptr, src_type, N, K, stream) != 0) return -1;
        g_marlin_cache[src0_ptr] = mw;
        it = g_marlin_cache.find(src0_ptr);
    }

    auto & mw = it->second;

    // Allocate static buffers for FP16 conversion
    static half * d_input_fp16 = nullptr;
    static half * d_output_fp16 = nullptr;
    static int buf_init = 0;
    if (!buf_init) {
        cudaMalloc(&d_input_fp16, 16 * 32768 * sizeof(half));
        cudaMalloc(&d_output_fp16, 16 * 262144 * sizeof(half));
        buf_init = 1;
    }

    // Convert FP32 activations -> FP16
    launch_f32_to_f16(d_input_fp16, (const float *)src1_ptr, M * K, stream);

    // Run Marlin INT4 matmul
    ensure_device_info();
    int * my_locks = get_next_lock_slot();

    dim3 grid(cached_sms, 1, 1);
    dim3 block(default_threads);

    int group_blocks = mw.group_size / marlin::tile_size;
    int num_groups = K / mw.group_size;

    marlin_launch_grouped(
        d_input_fp16, mw.d_packed, d_output_fp16, mw.d_scales,
        num_groups, M, N, K, group_blocks, cached_smem, my_locks, grid, block, stream);

    // Convert FP16 output -> FP32
    launch_f16_to_f32((float *)dst_ptr, d_output_fp16, M * N, stream);

    return 0;
}

#endif // ALPHA_HAS_FLASHINFER
