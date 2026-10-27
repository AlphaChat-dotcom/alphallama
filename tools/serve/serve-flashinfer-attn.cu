// FlashInfer Attention Kernel Integration
//
// Wraps FlashInfer's optimized BatchDecodeWithPagedKVCache kernel for use
// in the forward pass. Uses Tensor Core MMA, async memory pipeline, and
// split-K parallelism instead of a simple 32-thread scalar kernel.
//
// Instantiates templates for HEAD_DIM=64 and HEAD_DIM=128 (covers Qwen2.5, Qwen3, Llama, etc.)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <vector>

// FlashInfer headers (header-only, no external deps for decode attention)
#include "../../vendor/flashinfer/include/flashinfer/attention/decode.cuh"
#include "../../vendor/flashinfer/include/flashinfer/attention/default_decode_params.cuh"
#include "../../vendor/flashinfer/include/flashinfer/attention/variants.cuh"

using namespace flashinfer;

// Standard attention: no custom mask, no sliding window, no logits cap, no ALiBi
using StdAttention = DefaultAttention<false, false, false, false>;

// Per-head-dim workspace
// Pre-allocated GPU buffers for the scheduling metadata that FlashInfer needs
struct FIAttnWorkspace {
    int32_t * d_request_indices = nullptr;  // [max_padded_batch] identity mapping
    int32_t * d_kv_tile_indices = nullptr;  // [max_padded_batch] all zeros
    int32_t * d_o_indptr = nullptr;         // [max_padded_batch + 1] identity
    int32_t * d_kv_chunk_size = nullptr;    // single int32_t (large value)
    int max_padded_batch = 0;
    bool initialized = false;
};

static FIAttnWorkspace g_workspace;

static void ensure_workspace(int padded_batch) {
    // Pre-size to a generous floor so the workspace is allocated once and never reallocs mid-run:
    // a cudaFree/cudaMalloc here while a batched CUDA graph holds the old pointers would fault on
    // replay. This does not fix the batched-graph deadlock seen when serving more requests than
    // slots (batch sizes fluctuate as slots free and refill, and batched-graph capture/replay can
    // deadlock regardless of workspace stability); use ALPHA_NO_GRAPH_BATCH=1 for that case.
    // The arrays are tiny (int32), so over-allocating (covers max_seqs + n_ctx <= 16K) is ~free.
    // ALPHA_FI_WS_FLOOR overrides.
    static const int WS_FLOOR = []{ const char* e = getenv("ALPHA_FI_WS_FLOOR"); int v = e?atoi(e):16384; return v<1?1:v; }();
    if (padded_batch < WS_FLOOR) padded_batch = WS_FLOOR;
    if (g_workspace.initialized && g_workspace.max_padded_batch >= padded_batch) return;

    // Free old
    if (g_workspace.d_request_indices) cudaFree(g_workspace.d_request_indices);
    if (g_workspace.d_kv_tile_indices) cudaFree(g_workspace.d_kv_tile_indices);
    if (g_workspace.d_o_indptr) cudaFree(g_workspace.d_o_indptr);
    if (g_workspace.d_kv_chunk_size) cudaFree(g_workspace.d_kv_chunk_size);

    g_workspace.max_padded_batch = padded_batch;

    // Allocate
    cudaMalloc(&g_workspace.d_request_indices, padded_batch * sizeof(int32_t));
    cudaMalloc(&g_workspace.d_kv_tile_indices, padded_batch * sizeof(int32_t));
    cudaMalloc(&g_workspace.d_o_indptr, (padded_batch + 1) * sizeof(int32_t));
    cudaMalloc(&g_workspace.d_kv_chunk_size, sizeof(int32_t));

    // Fill identity: request_indices[i] = i, kv_tile_indices[i] = 0, o_indptr[i] = i
    std::vector<int32_t> h_indices(padded_batch);
    std::vector<int32_t> h_zeros(padded_batch, 0);
    std::vector<int32_t> h_indptr(padded_batch + 1);
    for (int i = 0; i < padded_batch; i++) {
        h_indices[i] = i;
        h_indptr[i] = i;
    }
    h_indptr[padded_batch] = padded_batch;

    cudaMemcpy(g_workspace.d_request_indices, h_indices.data(),
               padded_batch * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(g_workspace.d_kv_tile_indices, h_zeros.data(),
               padded_batch * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(g_workspace.d_o_indptr, h_indptr.data(),
               (padded_batch + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);

    // kv_chunk_size = large value (no partition)
    int32_t large_val = 1 << 20;  // 1M tokens max
    cudaMemcpy(g_workspace.d_kv_chunk_size, &large_val, sizeof(int32_t), cudaMemcpyHostToDevice);

    g_workspace.initialized = true;
    fprintf(stderr, "[fi-attn] workspace initialized: max_padded_batch=%d\n", padded_batch);
}

// Template instantiation for specific head dims

// Supported GQA group sizes for FlashInfer's template dispatch
static int fi_padded_group_size(int n_qo_heads, int n_kv_heads) {
    if (n_kv_heads == 0) return 0;
    int gs = n_qo_heads / n_kv_heads;
    if (gs == 1 || gs == 2 || gs == 3 || gs == 4 || gs == 7 || gs == 8) return gs;
    if (gs <= 4) return 4;
    if (gs <= 8) return 8;
    return 0;
}

// Pre-allocated Q padding buffer for non-standard GQA ratios
static half * d_q_padded_ = nullptr;
static half * d_o_padded_ = nullptr;
static int q_padded_size_ = 0;

static void ensure_q_padding(int padded_heads, int head_dim, int max_batch) {
    int needed = max_batch * padded_heads * head_dim;
    if (d_q_padded_ && q_padded_size_ >= needed) return;
    if (d_q_padded_) cudaFree(d_q_padded_);
    if (d_o_padded_) cudaFree(d_o_padded_);
    cudaMalloc(&d_q_padded_, needed * sizeof(half));
    cudaMalloc(&d_o_padded_, needed * sizeof(half));
    cudaMemset(d_q_padded_, 0, needed * sizeof(half));
    q_padded_size_ = needed;
}

// Simple CUDA kernel to scatter Q heads into padded layout
// For GQA 7->8: copy 7 heads per KV group, leave 8th as zero
__global__ void pad_q_heads_kernel(
    half * __restrict__ dst,        // [batch, padded_heads, head_dim]
    const half * __restrict__ src,  // [batch, real_heads, head_dim]
    int real_heads, int padded_heads, int head_dim,
    int real_group, int padded_group, int n_kv_heads)
{
    int b = blockIdx.x;
    int h = blockIdx.y;  // padded head index
    if (h >= padded_heads) return;

    int kv_head = h / padded_group;
    int local = h % padded_group;

    half * d = dst + ((int64_t)b * padded_heads + h) * head_dim;

    if (kv_head < n_kv_heads && local < real_group) {
        int src_head = kv_head * real_group + local;
        const half * s = src + ((int64_t)b * real_heads + src_head) * head_dim;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) d[i] = s[i];
    } else {
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) d[i] = __float2half(0.0f);
    }
}

// Reverse: extract real heads from padded output
__global__ void unpad_o_heads_kernel(
    half * __restrict__ dst,        // [batch, real_heads, head_dim]
    const half * __restrict__ src,  // [batch, padded_heads, head_dim]
    int real_heads, int padded_heads, int head_dim,
    int real_group, int padded_group, int n_kv_heads)
{
    int b = blockIdx.x;
    int h = blockIdx.y;  // real head index
    if (h >= real_heads) return;

    int kv_head = h / real_group;
    int local = h % real_group;
    int padded_h = kv_head * padded_group + local;

    const half * s = src + ((int64_t)b * padded_heads + padded_h) * head_dim;
    half * d = dst + ((int64_t)b * real_heads + h) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) d[i] = s[i];
}

template <uint32_t HEAD_DIM>
static cudaError_t launch_fi_decode_impl(
    half * output,       // [batch, n_qo_heads, head_dim]
    const half * q,      // [batch, n_qo_heads, head_dim]
    half * k_pool,       // [n_pages, page_size, n_kv_heads, head_dim]
    half * v_pool,       // same layout
    const int32_t * indptr,       // [batch + 1]
    const int32_t * indices,      // [total_pages]
    const int32_t * last_page_len, // [batch]
    int page_size,
    int n_qo_heads,
    int n_kv_heads,
    int batch_size,
    int padded_batch_size,
    int stride_page,
    int stride_n,
    int stride_h,
    cudaStream_t stream)
{
    ensure_workspace(padded_batch_size);

    int real_group = n_qo_heads / n_kv_heads;
    int padded_group = fi_padded_group_size(n_qo_heads, n_kv_heads);
    int padded_qo_heads = n_kv_heads * padded_group;
    bool needs_padding = (padded_group != real_group);

    // If GQA ratio needs padding, rearrange Q heads
    const half * q_ptr = q;
    half * o_ptr = output;
    if (needs_padding) {
        ensure_q_padding(padded_qo_heads, HEAD_DIM, padded_batch_size);
        dim3 grid(padded_batch_size, padded_qo_heads);
        int threads = (HEAD_DIM < 128) ? HEAD_DIM : 128;
        threads = ((threads + 31) / 32) * 32;
        pad_q_heads_kernel<<<grid, threads, 0, stream>>>(
            d_q_padded_, q, n_qo_heads, padded_qo_heads, HEAD_DIM,
            real_group, padded_group, n_kv_heads);
        q_ptr = d_q_padded_;
        o_ptr = d_o_padded_;
    }

    // Build FlashInfer paged_kv_t
    int64_t kv_strides[3] = {
        (int64_t)stride_page,
        (int64_t)stride_n,
        (int64_t)stride_h
    };

    paged_kv_t<half, int32_t> paged_kv(
        n_kv_heads, page_size, HEAD_DIM, batch_size,
        QKVLayout::kNHD,
        k_pool, v_pool, kv_strides,
        const_cast<int32_t*>(indices),
        const_cast<int32_t*>(indptr),
        const_cast<int32_t*>(last_page_len),
        nullptr
    );

    float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);

    BatchDecodeParams<half, half, half, int32_t> params(
        const_cast<half*>(q_ptr),
        nullptr,
        paged_kv,
        o_ptr,
        nullptr, nullptr,
        padded_qo_heads,
        padded_qo_heads * HEAD_DIM,
        HEAD_DIM,
        -1, 0.0f, sm_scale, 1.0f, 10000.0f
    );

    params.padded_batch_size = padded_batch_size;
    params.request_indices = g_workspace.d_request_indices;
    params.kv_tile_indices = g_workspace.d_kv_tile_indices;
    params.o_indptr = g_workspace.d_o_indptr;
    params.kv_chunk_size_ptr = g_workspace.d_kv_chunk_size;
    params.block_valid_mask = nullptr;
    params.partition_kv = false;

    cudaError_t err = BatchDecodeWithPagedKVCacheDispatched<
        HEAD_DIM,
        PosEncodingMode::kNone,
        StdAttention,
        BatchDecodeParams<half, half, half, int32_t>
    >(params, nullptr, nullptr, false, stream);

    // Extract real heads from padded output
    if (needs_padding && err == cudaSuccess) {
        dim3 grid2(padded_batch_size, n_qo_heads);
        int threads = (HEAD_DIM < 128) ? HEAD_DIM : 128;
        threads = ((threads + 31) / 32) * 32;
        unpad_o_heads_kernel<<<grid2, threads, 0, stream>>>(
            output, d_o_padded_, n_qo_heads, padded_qo_heads, HEAD_DIM,
            real_group, padded_group, n_kv_heads);
    }

    return err;
}

// C API

extern "C" int flashinfer_attn_supported(int n_qo_heads, int n_kv_heads, int head_dim) {
    if (head_dim != 64 && head_dim != 128 && head_dim != 256) return 0;
    return fi_padded_group_size(n_qo_heads, n_kv_heads) > 0 ? 1 : 0;
}

extern "C" cudaError_t launch_flashinfer_batch_decode(
    half * output, const half * q,
    half * k_pool, half * v_pool,
    const int32_t * indptr, const int32_t * indices, const int32_t * last_page_len,
    int page_size, int n_qo_heads, int n_kv_heads, int head_dim,
    int batch_size, int padded_batch_size,
    int stride_page, int stride_n, int stride_h,
    cudaStream_t stream)
{
    if (!fi_padded_group_size(n_qo_heads, n_kv_heads)) {
        fprintf(stderr, "[fi-attn] unsupported GQA group_size=%d/%d, use ALPHA_NO_FI_ATTN=1\n",
                n_qo_heads, n_kv_heads);
        return cudaErrorInvalidValue;
    }

    switch (head_dim) {
        case 64:
            return launch_fi_decode_impl<64>(
                output, q, k_pool, v_pool,
                indptr, indices, last_page_len,
                page_size, n_qo_heads, n_kv_heads, batch_size, padded_batch_size,
                stride_page, stride_n, stride_h, stream);
        case 128:
            return launch_fi_decode_impl<128>(
                output, q, k_pool, v_pool,
                indptr, indices, last_page_len,
                page_size, n_qo_heads, n_kv_heads, batch_size, padded_batch_size,
                stride_page, stride_n, stride_h, stream);
        case 256:
            return launch_fi_decode_impl<256>(
                output, q, k_pool, v_pool,
                indptr, indices, last_page_len,
                page_size, n_qo_heads, n_kv_heads, batch_size, padded_batch_size,
                stride_page, stride_n, stride_h, stream);
        default:
            fprintf(stderr, "[fi-attn] unsupported head_dim=%d (need 64 or 128)\n", head_dim);
            return cudaErrorInvalidValue;
    }
}
