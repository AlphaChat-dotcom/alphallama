#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_src1[nex_prev + itc] = it*sis1          + iex_used % nchannels_y;
        ids_dst [nex_prev + itc] = it*n_expert_used + iex_used;
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

// Multi-warp variant (ALPHA_MMID_MW) for any n_expert_used <= 32, padded to a power of two lanes per token. The
// one-warp kernel scans every token of the batch serially per expert, which dominates at prefill batch sizes.
// Here NW warps split the tokens into contiguous segments: pass 1 counts each segment's hits and lower-expert
// columns, a block prefix gives each warp its output offset, pass 2 writes ids_src1/ids_dst straight to their
// final positions. The token order per expert is unchanged, so the output is identical.
template <int npad, int NW>
__launch_bounds__(32*NW, 1)
static __global__ void mm_ids_helper_mw(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1) {
    constexpr int tpi = 32/npad;   // tokens per warp iteration
    const int expert = blockIdx.x;
    const int lane = threadIdx.x, w = threadIdx.y;
    int seg = (n_tokens + NW - 1)/NW;
    seg = (seg + tpi - 1)/tpi*tpi;
    const int t_begin = min(n_tokens, w*seg), t_end = min(n_tokens, t_begin + seg);
    const int iex = lane % npad;

    __shared__ int s_cnt[NW], s_prev[NW];

    // pass 1
    int cnt = 0, prev = 0;
    for (int it0 = t_begin; it0 < t_end; it0 += tpi) {
        const int it = it0 + lane/npad;
        const int e = iex < n_expert_used && it < t_end ? ids[it*si1 + iex] : INT_MAX;
        prev += e < expert;
        const int hit = warp_reduce_any<npad>(e == expert);
        cnt += (lane % npad == 0) ? hit : 0;
    }
    cnt  = warp_reduce_sum<32>(cnt);
    prev = warp_reduce_sum<32>(prev);
    if (lane == 0) { s_cnt[w] = cnt; s_prev[w] = prev; }
    __syncthreads();
    int off = 0, nex_prev = 0, total = 0;
#pragma unroll
    for (int k = 0; k < NW; ++k) {
        off      += k < w ? s_cnt[k] : 0;
        total    += s_cnt[k];
        nex_prev += s_prev[k];
    }

    // pass 2
    int it_compact = 0;
    for (int it0 = t_begin; it0 < t_end; it0 += tpi) {
        const int it = it0 + lane/npad;
        const int e = iex < n_expert_used && it < t_end ? ids[it*si1 + iex] : INT_MAX;
        const int iex_used = e == expert ? iex : -1;
        const int self = warp_reduce_any<npad>(iex_used != -1);
        int lower = 0;
#pragma unroll
        for (int o = npad; o < 32; o += npad) {
            const int tmp = __shfl_up_sync(0xFFFFFFFF, self, o, 32);
            if (lane >= o) lower += tmp;
        }
        if (iex_used != -1) {
            const int pos = nex_prev + off + it_compact + lower;
            ids_src1[pos] = it*sis1          + iex_used % nchannels_y;
            ids_dst [pos] = it*n_expert_used + iex_used;
        }
        it_compact += __shfl_sync(0xFFFFFFFF, lower + self, 31, 32);
    }

    if (lane != 0 || w != 0) {
        return;
    }
    expert_bounds[expert] = nex_prev;
    if (expert == static_cast<int>(gridDim.x) - 1) {
        expert_bounds[gridDim.x] = nex_prev + total;
    }
}

template <int npad>
static void launch_mm_ids_helper_mw(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    constexpr int NW = 8;
    mm_ids_helper_mw<npad, NW><<<dim3(n_experts, 1, 1), dim3(32, NW, 1), 0, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used, nchannels_y, si1, sis1);
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    static const bool mw = [] { const char * e = getenv("ALPHA_MMID_MW"); return !e || atoi(e) != 0; }();
    if (mw && ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size == 32 && n_tokens >= 256 && n_expert_used <= 32) {
        if      (n_expert_used <=  2) launch_mm_ids_helper_mw< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
        else if (n_expert_used <=  4) launch_mm_ids_helper_mw< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
        else if (n_expert_used <=  8) launch_mm_ids_helper_mw< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
        else if (n_expert_used <= 16) launch_mm_ids_helper_mw<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
        else                          launch_mm_ids_helper_mw<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
        return;
    }
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
    }
}
