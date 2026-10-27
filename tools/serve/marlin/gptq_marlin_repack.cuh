// Standalone GPTQ→Marlin repack kernel (no PyTorch/sgl_kernel deps)
// Ported from SGLang/vLLM, original by IST-DASLab/Elias Frantar
// Apache 2.0 license

#pragma once
#include "marlin.cuh"

namespace marlin {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
template <int const num_threads, int const num_bits, bool const has_perm>
__global__ void gptq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr,
    uint32_t const* __restrict__ perm_ptr,
    uint32_t* __restrict__ out_ptr,
    int size_k, int size_n) { return; }
#else
template <int const num_threads, int const num_bits, bool const has_perm>
__global__ void gptq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr,
    uint32_t const* __restrict__ perm_ptr,
    uint32_t* __restrict__ out_ptr,
    int size_k, int size_n)
{
  constexpr int pack_factor = 32 / num_bits;
  int k_tiles = size_k / tile_k_size;
  int n_tiles = size_n / tile_n_size;
  int block_k_tiles = div_ceil(k_tiles, gridDim.x);

  auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) return;
  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  auto wait_for_stage = [&]() {
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];
  constexpr int perm_size = tile_k_size / 4;
  int4* sh_perm_ptr = sh;
  int4* sh_pipe_ptr = sh_perm_ptr;
  if constexpr (has_perm) { sh_pipe_ptr += perm_size; }

  constexpr int tile_ints = tile_k_size / pack_factor;
  constexpr int stage_n_threads = tile_n_size / 4;
  constexpr int stage_k_threads = has_perm ? tile_k_size : tile_ints;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) { cp_async_fence(); return; }
    int first_n = n_tile_id * tile_n_size;
    int4* sh_ptr = sh_pipe_ptr + stage_size * pipe;

    if constexpr (!has_perm) {
      if (threadIdx.x < stage_size) {
        auto k_id = threadIdx.x / stage_n_threads;
        auto n_id = threadIdx.x % stage_n_threads;
        int first_k = k_tile_id * tile_k_size;
        int first_k_packed = first_k / pack_factor;
        cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                  reinterpret_cast<int4 const*>(
                    &(b_q_weight_ptr[(first_k_packed + k_id) * size_n + first_n + (n_id * 4)])));
      }
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
    constexpr int sh_stride = 64;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    uint32_t vals[8];
    uint32_t b1_vals[tile_ints];
    uint32_t b2_vals[tile_ints];

#pragma unroll
    for (int i = 0; i < tile_ints; i++) {
      b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i];
      b2_vals[i] = sh_stage_int_ptr[cur_n + 8 + sh_stride * i];
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
      int cur_elem = tc_row + tc_offsets[i];
      int cur_int = cur_elem / pack_factor;
      int cur_pos = cur_elem % pack_factor;
      vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
      vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
    }

    constexpr int tile_size = tile_k_size * tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;

    if constexpr (num_bits == 4) {
      constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }
      out_ptr[out_offset + th_id * 4 + warp_id] = res;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }
    wait_for_stage();
  };

  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;
    start_pipes(k_tile_id, n_tile_id);
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

// C-linkage launcher
inline void launch_gptq_marlin_repack(
    const uint32_t * b_q_weight, uint32_t * out,
    int size_k, int size_n, cudaStream_t stream)
{
    int dev = 0;
    cudaGetDevice(&dev);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    auto * kernel = gptq_marlin_repack_kernel<repack_threads, 4, false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    kernel<<<num_sms, repack_threads, max_shared_mem, stream>>>(
        b_q_weight, nullptr, out, size_k, size_n);
}

}  // namespace marlin
