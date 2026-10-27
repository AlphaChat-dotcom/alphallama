#include "argsort.cuh"
#include "top-k.cuh"

#include <algorithm>
#include <vector>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

// Top-k by radix select, one block per row (ALPHA_TOPK_RADIX, default on): four 8-bit histogram passes over the
// order-preserving uint image of each float find the k-th largest key T; every key > T is emitted, then the
// lowest-indexed keys == T until k are taken. This is the same set a stable descending sort's first k gives, so
// callers that use the indices as a set (QSA masks / gathers) see identical results, without sorting each row in full.
#include <cub/block/block_scan.cuh>

static __device__ __forceinline__ unsigned int topk_key(const float f) {
    const unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// key sources: a plain row, or a QSA cell score computed on the fly (ALPHA_QSA_TOPK_EXPAND): the score of the
// cell's block plus the cell's mask value, the same float add a materialised expand + add performs
struct topk_src_row {
    const float * x; int64_t stride;
    __device__ topk_src_row row(const int r) const { return { x + r*stride, stride }; }
    __device__ float operator()(const int i) const { return x[i]; }
};
template <bool MF16>
struct topk_src_expand {
    const float * sb; int64_t sb_stride; const int32_t * blk; const char * mrow; int64_t m_stride;
    __device__ topk_src_expand row(const int r) const { return { sb + r*sb_stride, sb_stride, blk, mrow + r*m_stride, m_stride }; }
    __device__ float operator()(const int i) const {
        const float m = MF16 ? __half2float(((const half *) mrow)[i]) : ((const float *) mrow)[i];
        return sb[blk[i]] + m;
    }
};

template <int BS, typename SRC>
static __global__ void topk_radix_select_src(const SRC src, int * __restrict__ dst, const int ncols, const int k) {
    const SRC x = src.row(blockIdx.x);
    int * out = dst + (int64_t) blockIdx.x * k;
    const int tid = threadIdx.x;

    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix, s_mask;
    __shared__ int s_remaining;
    if (tid == 0) { s_prefix = 0; s_mask = 0; s_remaining = k; }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8*pass;
        for (int b = tid; b < 256; b += BS) hist[b] = 0;
        __syncthreads();
        const unsigned int prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < ncols; i += BS) {
            const unsigned int key = topk_key(x(i));
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int rem = s_remaining;
            int acc = 0;
            int sel = 0;
            for (int b = 255; b >= 0; --b) {
                const int h = (int) hist[b];
                if (acc + h >= rem) { sel = b; rem -= acc; break; }
                acc += h;
            }
            s_prefix    = prefix | ((unsigned int) sel << shift);
            s_mask      = mask | (0xFFu << shift);
            s_remaining = rem;
        }
        __syncthreads();
    }
    const unsigned int T = s_prefix;
    const int n_ties = s_remaining;     // keys == T to take
    const int n_gt   = k - n_ties;      // keys > T (all of them)

    // contiguous chunk per thread, so ties are taken in index order
    const int chunk = (ncols + BS - 1)/BS;
    const int i0 = min(ncols, tid*chunk), i1 = min(ncols, i0 + chunk);
    int c_gt = 0, c_eq = 0;
    for (int i = i0; i < i1; ++i) {
        const unsigned int key = topk_key(x(i));
        c_gt += key > T;
        c_eq += key == T;
    }
    typedef cub::BlockScan<int, BS> scan_t;
    __shared__ typename scan_t::TempStorage tmp;
    int b_gt, b_eq;
    scan_t(tmp).ExclusiveSum(c_gt, b_gt);
    __syncthreads();
    scan_t(tmp).ExclusiveSum(c_eq, b_eq);
    for (int i = i0; i < i1; ++i) {
        const unsigned int key = topk_key(x(i));
        if (key > T) {
            out[b_gt++] = i;
        } else if (key == T) {
            if (b_eq < n_ties) out[n_gt + b_eq] = i;
            b_eq++;
        }
    }
}


void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    if (dst->src[1]) {   // QSA expand: src0 = block scores [n_blocks, n_rows], src1 = cell->block, src2 = mask
        const ggml_tensor * blk = dst->src[1], * msk = dst->src[2];
        const int n_cells = (int) blk->ne[0], nr = (int) src0->ne[1], kk = (int) dst->ne[0];
        if (msk->type == GGML_TYPE_F16) {
            topk_radix_select_src<1024><<<nr, 1024, 0, stream>>>(topk_src_expand<true>{ src0_d, src0->ne[0],
                (const int32_t *) blk->data, (const char *) msk->data, (int64_t) msk->nb[1] }, dst_d, n_cells, kk);
        } else {
            topk_radix_select_src<1024><<<nr, 1024, 0, stream>>>(topk_src_expand<false>{ src0_d, src0->ne[0],
                (const int32_t *) blk->data, (const char *) msk->data, (int64_t) msk->nb[1] }, dst_d, n_cells, kk);
        }
        return;
    }

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
    {
        static const bool radix = [] { const char * e = getenv("ALPHA_TOPK_RADIX"); return !e || atoi(e) != 0; }();
        if (radix && ncols > 1024 && k <= ncols && ncols <= INT32_MAX && nrows <= INT32_MAX) {
            topk_radix_select_src<1024><<<(unsigned) nrows, 1024, 0, stream>>>(topk_src_row{ src0_d, ncols }, dst_d, (int) ncols, (int) k);
#if defined(GGML_CUDA_USE_CUB) && !defined(CUB_TOP_K_AVAILABLE)
            // ALPHA_TOPK_CHECK=1 (debug, syncs): compare the selected set against the argsort path, row by row
            static const bool check = [] { const char * e = getenv("ALPHA_TOPK_CHECK"); return e && atoi(e) != 0; }();
            if (check) {
                ggml_cuda_pool_alloc<int> ref(pool, ncols * nrows);
                argsort_f32_i32_cuda_cub(pool, src0_d, ref.get(), ncols, nrows, GGML_SORT_ORDER_DESC, stream);
                std::vector<int> a((size_t) (k*nrows)), b((size_t) (ncols*nrows));
                CUDA_CHECK(cudaMemcpyAsync(a.data(), dst_d, a.size()*sizeof(int), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(b.data(), ref.get(), b.size()*sizeof(int), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                static long long n_rows = 0, n_bad = 0;
                for (int64_t r = 0; r < nrows; ++r) {
                    std::vector<int> x(a.begin() + r*k, a.begin() + (r + 1)*k), y(b.begin() + r*ncols, b.begin() + r*ncols + k);
                    std::sort(x.begin(), x.end()); std::sort(y.begin(), y.end());
                    n_rows++; n_bad += x != y;
                }
                if (n_bad || (n_rows % 4096) < (long long) nrows) {
                    fprintf(stderr, "[topk-check] %lld rows checked, %lld differ (ncols %lld, k %lld)\n", n_rows, n_bad, (long long) ncols, (long long) k);
                }
            }
#endif
            return;
        }
    }
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    if (shared_mem > max_shared_mem || ncols > 1024) {
        argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
