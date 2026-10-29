// SPDX-License-Identifier: Apache-2.0
// AlphaServe CUDA Kernels
//
// Lightweight kernels for the FlashInfer forward pass engine:
//   - Paged decode attention (simple version, FlashInfer swap-in later)
//   - RMSNorm
//   - RoPE (rotary position embedding)
//   - SiLU activation x element-wise multiply
//   - Token embedding lookup
//   - Residual addition
//
// These replace ggml's kernel dispatch with direct CUDA launches.
// Designed for FP16 (half) compute throughout.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// Paged Decode Attention
// Simple batched decode attention with paged KV cache.
// One block per (batch, head) pair. Iterates over KV pages sequentially.
// Uses online softmax (numerically stable, single pass for short contexts).
// TODO: swap with FlashInfer's optimized kernel for production.

__global__ void paged_decode_attention_kernel(
    half * __restrict__ output,        // [batch, n_heads, head_dim]
    const half * __restrict__ q,       // [batch, n_heads, head_dim]
    const half * __restrict__ k_pool,  // [n_pages, page_size, n_kv_heads, head_dim]
    const half * __restrict__ v_pool,  // same layout
    const int32_t * __restrict__ indptr,       // [batch + 1]
    const int32_t * __restrict__ indices,      // [total_pages]
    const int32_t * __restrict__ last_page_len, // [batch]
    int page_size,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int stride_page,   // page_size * n_kv_heads * head_dim
    int stride_n,      // n_kv_heads * head_dim
    int stride_h,      // head_dim
    float scale,       // 1/sqrt(head_dim)
    int window)        // sliding-window width (>0 -> attend only to the last `window` KV positions; <=0 -> full causal)
{
    // FlashDecoding-style: block = one (batch,head); its warps split the KV range so many KV
    // positions are processed in parallel (one warp per block would serialise the KV loop at low
    // occupancy). Each warp keeps its own online-softmax partial, then the block combines them.
    // Each lane holds ELEMS=ceil(head_dim/32) of the head vector.
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;
    const int ELEMS = (head_dim + 31) / 32;

    const half * q_ptr = q + (int64_t)batch_idx * n_heads * head_dim + head_idx * head_dim;
    float q_val[8];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) { int d = lane + e * 32; q_val[e] = (d < head_dim) ? __half2float(q_ptr[d]) : 0.0f; }

    int page_start = indptr[batch_idx];
    int page_end   = indptr[batch_idx + 1];
    int lpl        = last_page_len[batch_idx];
    int total_kv   = (page_end - page_start - 1) * page_size + lpl;
    int min_pos    = (window > 0 && total_kv - window > 0) ? (total_kv - window) : 0;

    // Each warp handles logical KV positions [min_pos+warp, total_kv) strided by n_warps.
    float max_s = -1e30f, sum_e = 0.0f, acc[8] = {0};
    for (int pos = min_pos + warp; pos < total_kv; pos += n_warps) {
        int page_id = indices[page_start + pos / page_size];
        int t = pos % page_size;
        const half * k_ptr = k_pool + (int64_t)page_id * stride_page + t * stride_n + kv_head_idx * stride_h;
        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) { int d = lane + e * 32; if (d < head_dim) dot += q_val[e] * __half2float(k_ptr[d]); }
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
        float score = __shfl_sync(0xffffffff, dot, 0) * scale;   // broadcast lane0's full sum
        float old = max_s; if (score > max_s) max_s = score;
        float eo = __expf(old - max_s), en = __expf(score - max_s);
        sum_e = sum_e * eo + en;
        const half * v_ptr = v_pool + (int64_t)page_id * stride_page + t * stride_n + kv_head_idx * stride_h;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) { int d = lane + e * 32; float v = (d < head_dim) ? __half2float(v_ptr[d]) : 0.0f; acc[e] = acc[e] * eo + en * v; }
    }

    // Combine warp partials (flash-attention rescale)
    __shared__ float sh_max[32], sh_sum[32];
    extern __shared__ float sh_acc[];   // [n_warps * head_dim]
    if (lane == 0) { sh_max[warp] = max_s; sh_sum[warp] = sum_e; }
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) { int d = lane + e * 32; if (d < head_dim) sh_acc[warp * head_dim + d] = acc[e]; }
    __syncthreads();

    float gmax = -1e30f;
    for (int w = 0; w < n_warps; w++) gmax = fmaxf(gmax, sh_max[w]);
    float gsum = 0.0f;
    for (int w = 0; w < n_warps; w++) gsum += sh_sum[w] * __expf(sh_max[w] - gmax);
    float inv = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;

    half * out_ptr = output + (int64_t)batch_idx * n_heads * head_dim + head_idx * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float o = 0.0f;
        for (int w = 0; w < n_warps; w++) o += sh_acc[w * head_dim + d] * __expf(sh_max[w] - gmax);
        out_ptr[d] = __float2half(o * inv);
    }
}

extern "C" void launch_paged_decode_attention(
    half * output, const half * q,
    const half * k_pool, const half * v_pool,
    const int32_t * indptr, const int32_t * indices, const int32_t * last_page_len,
    int page_size, int n_heads, int n_kv_heads, int head_dim,
    int stride_page, int stride_n, int stride_h,
    int batch_size, int window, cudaStream_t stream)
{
    float scale = 1.0f / sqrtf((float)head_dim);

    // Several warps per block so the KV loop runs in parallel (FlashDecoding within a block).
    // ALPHA_ATTN_WARPS overrides the warp count (default 4). Dynamic shared = n_warps*head_dim
    // floats for the combine.
    static int env_w = getenv("ALPHA_ATTN_WARPS") ? atoi(getenv("ALPHA_ATTN_WARPS")) : 4;
    int n_warps = env_w < 1 ? 1 : (env_w > 32 ? 32 : env_w);
    int threads = n_warps * 32;
    size_t smem = (size_t)n_warps * head_dim * sizeof(float);
    dim3 grid(batch_size, n_heads);

    paged_decode_attention_kernel<<<grid, threads, smem, stream>>>(
        output, q, k_pool, v_pool,
        indptr, indices, last_page_len,
        page_size, n_heads, n_kv_heads, head_dim,
        stride_page, stride_n, stride_h, scale, window);
}

// Paged Prefill Attention
// Multiple query tokens attending to paged KV cache with causal masking.
// Used for prefix-cached prefill: S suffix queries attend to M+S keys
// where M prefix keys are already in the paged cache.
// Grid: (n_queries, n_heads), Block: 32 threads (one warp).
// Causal: query at absolute position q_pos attends to keys 0..q_pos.

__global__ void paged_prefill_attention_kernel(
    half * __restrict__ output,        // [n_queries, n_heads, head_dim]
    const half * __restrict__ q,       // [n_queries, n_heads, head_dim]
    const half * __restrict__ k_pool,
    const half * __restrict__ v_pool,
    const int32_t * __restrict__ indptr,
    const int32_t * __restrict__ indices,
    const int32_t * __restrict__ last_page_len,
    int page_size, int n_heads, int n_kv_heads, int head_dim,
    int stride_page, int stride_n, int stride_h,
    float scale,
    int n_queries,
    int prefix_len)     // absolute position offset for suffix queries
{
    const int query_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    if (query_idx >= n_queries) return;

    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);

    // Causal limit: this query at absolute position (prefix_len + query_idx)
    // can attend to keys at positions 0..(prefix_len + query_idx) inclusive
    const int n_kv_limit = prefix_len + query_idx + 1;

    const half * q_ptr = q + query_idx * n_heads * head_dim + head_idx * head_dim;

    float q_val[8];
    int elems_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
    if (elems_per_thread > 8) elems_per_thread = 8;
    for (int e = 0; e < elems_per_thread; e++) {
        int d = threadIdx.x + e * blockDim.x;
        q_val[e] = (d < head_dim) ? __half2float(q_ptr[d]) : 0.0f;
    }

    int page_start = indptr[0];
    int page_end = indptr[1];
    int lpl = last_page_len[0];

    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float acc[8] = {0};

    int kv_seen = 0;
    for (int p = page_start; p < page_end && kv_seen < n_kv_limit; p++) {
        int page_id = indices[p];
        int n_in_page = (p == page_end - 1) ? lpl : page_size;

        for (int t = 0; t < n_in_page && kv_seen < n_kv_limit; t++) {
            const half * k_ptr = k_pool + page_id * stride_page + t * stride_n + kv_head_idx * stride_h;

            float dot = 0.0f;
            for (int e = 0; e < elems_per_thread; e++) {
                int d = threadIdx.x + e * blockDim.x;
                if (d < head_dim) dot += q_val[e] * __half2float(k_ptr[d]);
            }
            for (int offset = warpSize / 2; offset > 0; offset /= 2)
                dot += __shfl_down_sync(0xffffffff, dot, offset);

            float score = 0.0f;
            if (threadIdx.x == 0) score = dot * scale;
            score = __shfl_sync(0xffffffff, score, 0);

            float old_max = max_score;
            if (score > max_score) max_score = score;
            float exp_old = expf(old_max - max_score);
            float exp_new = expf(score - max_score);
            sum_exp = sum_exp * exp_old + exp_new;

            const half * v_ptr = v_pool + page_id * stride_page + t * stride_n + kv_head_idx * stride_h;
            for (int e = 0; e < elems_per_thread; e++) {
                int d = threadIdx.x + e * blockDim.x;
                float v = (d < head_dim) ? __half2float(v_ptr[d]) : 0.0f;
                acc[e] = acc[e] * exp_old + exp_new * v;
            }
            kv_seen++;
        }
    }

    half * out_ptr = output + query_idx * n_heads * head_dim + head_idx * head_dim;
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int e = 0; e < elems_per_thread; e++) {
        int d = threadIdx.x + e * blockDim.x;
        if (d < head_dim) out_ptr[d] = __float2half(acc[e] * inv_sum);
    }
}

extern "C" void launch_paged_prefill_attention(
    half * output, const half * q,
    const half * k_pool, const half * v_pool,
    const int32_t * indptr, const int32_t * indices, const int32_t * last_page_len,
    int page_size, int n_heads, int n_kv_heads, int head_dim,
    int stride_page, int stride_n, int stride_h,
    int n_queries, int prefix_len, cudaStream_t stream)
{
    if (n_queries <= 0) return;
    float scale = 1.0f / sqrtf((float)head_dim);
    int threads = 32;
    dim3 grid(n_queries, n_heads);

    paged_prefill_attention_kernel<<<grid, threads, 0, stream>>>(
        output, q, k_pool, v_pool,
        indptr, indices, last_page_len,
        page_size, n_heads, n_kv_heads, head_dim,
        stride_page, stride_n, stride_h, scale,
        n_queries, prefix_len);
}

// Batch Argmax (GPU Sampling)
// One block per sequence. Parallel reduction finds argmax of vocab_size logits.
// Replaces CPU: copy 4.6MB -> iterate 151K floats -> copy 32 bytes back.
// New: one kernel -> copy 32 bytes back.

// Map a float to a uint whose unsigned ordering matches float ordering (monotonic key).
__device__ __forceinline__ unsigned int order_float(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Full-GPU argmax: grid (blocks_per_row, batch_size). Each block reduces a chunk of its row and
// combines into scratch[row] via a packed 64-bit atomicMax. High 32 bits = order_float(val); low
// 32 bits = (vocab-1-idx) so the smallest index wins ties (first maximum). Several blocks per row
// keep a large vocab from running on a single SM.
__global__ void argmax_reduce_kernel(
    unsigned long long * __restrict__ scratch,  // [batch_size], pre-zeroed
    const float * __restrict__ logits, int vocab_size)
{
    const int row = blockIdx.y;
    const float * r = logits + (int64_t)row * vocab_size;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    float best_val = -1e30f; int best_idx = 0;
    for (int v = gid; v < vocab_size; v += stride) {
        float val = r[v];
        if (val > best_val) { best_val = val; best_idx = v; }
    }
    for (int o = warpSize / 2; o > 0; o >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, best_val, o);
        int   oi = __shfl_down_sync(0xffffffff, best_idx, o);
        if (ov > best_val) { best_val = ov; best_idx = oi; }
    }
    __shared__ float s_v[32]; __shared__ int s_i[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int wid  = threadIdx.x / warpSize;
    if (lane == 0) { s_v[wid] = best_val; s_i[wid] = best_idx; }
    __syncthreads();
    if (wid == 0) {
        const int nw = blockDim.x / warpSize;
        best_val = (lane < nw) ? s_v[lane] : -1e30f;
        best_idx = (lane < nw) ? s_i[lane] : 0;
        for (int o = warpSize / 2; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, best_val, o);
            int   oi = __shfl_down_sync(0xffffffff, best_idx, o);
            if (ov > best_val) { best_val = ov; best_idx = oi; }
        }
        if (lane == 0) {
            unsigned long long packed =
                ((unsigned long long)order_float(best_val) << 32) |
                (unsigned int)(vocab_size - 1 - best_idx);
            atomicMax(scratch + row, packed);
        }
    }
}

__global__ void argmax_extract_kernel(
    int32_t * __restrict__ out_ids, const unsigned long long * __restrict__ scratch,
    int batch_size, int vocab_size)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    unsigned int low = (unsigned int)(scratch[b] & 0xffffffffu);
    out_ids[b] = vocab_size - 1 - (int)low;
}

extern "C" void launch_batch_argmax(
    int32_t * out_ids, const float * logits,
    int batch_size, int vocab_size, cudaStream_t stream)
{
    // Persistent scratch, allocated once at a fixed max on the first (eager) call and never freed/
    // realloced: a CUDA-graph capture bakes this pointer in, so it must stay stable across replays.
    static unsigned long long * scratch = nullptr;
    const int SCRATCH_MAX = 1024;
    if (!scratch) cudaMalloc(&scratch, (size_t)SCRATCH_MAX * sizeof(unsigned long long));
    if (batch_size > SCRATCH_MAX) return;  // never in practice (batch <= n_slots)
    cudaMemsetAsync(scratch, 0, (size_t)batch_size * sizeof(unsigned long long), stream);
    const int threads = 256;
    int bpr = (vocab_size + threads - 1) / threads;
    if (bpr > 256) bpr = 256;  // cap: 256 blocks x 256 threads saturates the GPU for the reduce
    dim3 grid(bpr, batch_size);
    argmax_reduce_kernel<<<grid, threads, 0, stream>>>(scratch, logits, vocab_size);
    argmax_extract_kernel<<<(batch_size + 63) / 64, 64, 0, stream>>>(out_ids, scratch, batch_size, vocab_size);
}

// Bias Add: out[batch, dim] += bias[dim]
__global__ void add_bias_kernel(half * __restrict__ out, const half * __restrict__ bias, int dim, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim;
    if (idx < total) {
        int d = idx % dim;
        out[idx] = __hadd(out[idx], bias[d]);
    }
}

extern "C" void launch_add_bias(half * out, const half * bias, int dim, int batch, cudaStream_t stream) {
    int n = batch * dim;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(out, bias, dim, batch);
}

// F32 -> FP16 Conversion

// int64 count + grid-stride: MoE expert stacks can exceed 2^31 elements.
__global__ void f32_to_f16_kernel(half * __restrict__ out, const float * __restrict__ inp, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        out[idx] = __float2half(inp[idx]);
    }
}

extern "C" void launch_f32_to_f16(half * out, const float * inp, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int64_t want = (n + threads - 1) / threads;
    int blocks = (want > 1048576) ? 1048576 : (int)want;
    f32_to_f16_kernel<<<blocks, threads, 0, stream>>>(out, inp, n);
}

__global__ void f16_to_f32_kernel(float * __restrict__ out, const half * __restrict__ inp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(inp[i]);
}

extern "C" void launch_f16_to_f32(float * out, const half * inp, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    f16_to_f32_kernel<<<blocks, threads, 0, stream>>>(out, inp, n);
}

// Q4_K Dequantization
// Converts Q4_K blocks to FP16. One CUDA block per Q4_K block (256 elements).
// Uses 32 threads, matching the layout of ggml's own dequantize_block_q4_K.
//
// Q4_K block layout (QK_K = 256 elements per block):
//   dm      : half2  - d = __low2half(dm), dmin = __high2half(dm)
//   scales  : uint8_t[12] - 8 sub-block scales + mins packed in 6 bits each
//   qs      : uint8_t[128] - 256 values packed as 4 bits each
//
// Each thread handles 4 consecutive quants at two offsets (+0 and +32),
// giving 32 threads x 8 outputs = 256 outputs per block.
//
// Scale/min unpacking (get_scale_min_k4):
//   j < 4  -> sc = scales[j] & 63,       m = scales[j+4] & 63
//   j >= 4 -> sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
//             m  = (scales[j+4] >>  4) | ((scales[j-0] >> 6) << 4)

#define Q4K_QK_K     256
#define Q4K_K_SCALE  12

// Packed Q4_K block as seen by the kernel (matches block_q4_K with CUDA half2).
// We declare our own to avoid pulling in ggml-common.h internals.
struct AlphaQ4KBlock {
    half2    dm;                    // d = __low2half(dm), dmin = __high2half(dm)
    uint8_t  scales[Q4K_K_SCALE];  // 6-bit packed scales and mins (12 bytes)
    uint8_t  qs[Q4K_QK_K / 2];    // 4-bit quants (128 bytes)
};
// Must be 144 bytes: 4 (half2) + 12 (scales) + 128 (qs)
static_assert(sizeof(AlphaQ4KBlock) == 144, "AlphaQ4KBlock size mismatch with block_q4_K");

static inline __device__ void alpha_get_scale_min_k4(int j, const uint8_t * q, uint8_t & sc, uint8_t & m) {
    if (j < 4) {
        sc = q[j]   & 63;
        m  = q[j+4] & 63;
    } else {
        sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m  = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// One block per Q4_K block, 32 threads per block.
// Each thread processes n=4 quants at two sub-block offsets -> 8 outputs.
// Total: 32 x 8 = 256 outputs = QK_K.
__global__ void dequant_q4k_kernel(half * __restrict__ out, const void * __restrict__ src, int /*n_blocks*/) {
    const AlphaQ4KBlock * x = (const AlphaQ4KBlock *)src;
    const int64_t i   = blockIdx.x;   // block index
    const int64_t tid = threadIdx.x;  // 0..31
    const int64_t il  = tid / 8;      // 0..3  - which sub-block pair (64 elems each)
    const int64_t ir  = tid % 8;      // 0..7  - position within sub-block
    const int     n   = 4;            // quants per thread per offset

    // Output pointer for this thread: base of this block + sub-block offset
    half * y = out + i * Q4K_QK_K + 64 * il + n * ir;

    // Extract d and dmin from the packed half2 field.
    // __low2half(half2) returns the low (first) half; __high2half returns the high (second) half.
    const float dall = __half2float(__low2half(x[i].dm));
    const float dmin = __half2float(__high2half(x[i].dm));

    const uint8_t * q = x[i].qs + 32 * il + n * ir;

    // Each sub-block pair covers indices is and is+1 in the scales array
    const int64_t is = 2 * il;
    uint8_t sc, m;

    alpha_get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;

    alpha_get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    for (int l = 0; l < n; ++l) {
        y[l +  0] = __float2half(d1 * (float)(q[l] & 0xF) - m1);
        y[l + 32] = __float2half(d2 * (float)(q[l] >>  4) - m2);
    }
}

extern "C" void launch_dequant_q4k(half * out, const void * src, int n_elements, cudaStream_t stream) {
    // One block per Q4_K super-block (256 elements), 32 threads each
    int n_blocks = n_elements / Q4K_QK_K;
    if (n_blocks <= 0) return;
    dequant_q4k_kernel<<<n_blocks, 32, 0, stream>>>(out, src, n_blocks);
}

// Deinterleave QKV
// Splits fused QKV output [batch, q_dim+kv_dim+kv_dim] into separate Q, K, V buffers.

__global__ void deinterleave_qkv_kernel(
    half * __restrict__ q_out,    // [batch, q_dim]
    half * __restrict__ k_out,    // [batch, kv_dim]
    half * __restrict__ v_out,    // [batch, kv_dim]
    const half * __restrict__ qkv, // [batch, qkv_dim]
    int q_dim, int kv_dim, int qkv_dim, int batch_size)
{
    int b = blockIdx.x;
    if (b >= batch_size) return;

    const half * src = qkv + (int64_t)b * qkv_dim;
    half * dq = q_out + (int64_t)b * q_dim;
    half * dk = k_out + (int64_t)b * kv_dim;
    half * dv = v_out + (int64_t)b * kv_dim;

    for (int i = threadIdx.x; i < q_dim; i += blockDim.x) dq[i] = src[i];
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) dk[i] = src[q_dim + i];
    if (dv) { for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) dv[i] = src[q_dim + kv_dim + i]; }
}

extern "C" void launch_deinterleave_qkv(
    half * q, half * k, half * v, const half * qkv,
    int q_dim, int kv_dim, int batch_size, cudaStream_t stream)
{
    int qkv_dim = q_dim + 2 * kv_dim;
    int threads = (q_dim < 256) ? q_dim : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    deinterleave_qkv_kernel<<<batch_size, threads, 0, stream>>>(
        q, k, v, qkv, q_dim, kv_dim, qkv_dim, batch_size);
}

// KV Scatter Write
// GPU kernel to write K/V data into paged cache at pre-computed offsets.
// Replaces per-layer cudaMemcpyAsync for CUDA graph compatibility.

__global__ void scatter_kv_kernel(
    half * __restrict__ pool,
    const half * __restrict__ src,
    const int64_t * __restrict__ offsets,
    int kv_dim,
    int batch_size)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int64_t dst_offset = offsets[batch_idx];
    const half * s = src + (int64_t)batch_idx * kv_dim;
    half * d = pool + dst_offset;

    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        d[i] = s[i];
    }
}

extern "C" void launch_scatter_kv(
    half * pool, const half * src, const int64_t * offsets,
    int kv_dim, int batch_size, cudaStream_t stream)
{
    if (batch_size <= 0) return;
    int threads = (kv_dim < 256) ? kv_dim : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    scatter_kv_kernel<<<batch_size, threads, 0, stream>>>(pool, src, offsets, kv_dim, batch_size);
}

// RMSNorm
// out[i] = (inp[i] / rms) * weight[i]
// rms = sqrt(mean(inp^2) + eps)

__global__ void rms_norm_kernel(
    half * __restrict__ out,
    const half * __restrict__ inp,
    const half * __restrict__ weight,
    int n,
    float eps)
{
    const int row = blockIdx.x;
    const half * x = inp + row * n;
    half * y = out + row * n;

    // Compute sum of squares (use float for precision)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __half2float(x[i]);
        sum_sq += v * v;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane < blockDim.x / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    // Broadcast rms_inv
    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) {
        s_rms_inv = rsqrtf(sum_sq / (float)n + eps);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;

    // Apply normalization + weight
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __half2float(x[i]) * rms_inv;
        float w = __half2float(weight[i]);
        y[i] = __float2half(v * w);
    }
}

extern "C" void launch_rms_norm(
    half * out, const half * inp, const half * weight,
    int n, int batch, float eps, cudaStream_t stream)
{
    int threads = (n < 1024) ? n : 1024;
    // Round up to multiple of 32
    threads = ((threads + 31) / 32) * 32;
    rms_norm_kernel<<<batch, threads, 0, stream>>>(out, inp, weight, n, eps);
}

// Per-head QK-norm (Qwen3/3.5)
// Applies RMSNorm independently to each head's head_dim slice of a buffer, in place.
// grid=(batch, n_heads); one block per (token, head). Buffer layout per token is
// [token_stride] with the target region starting at base_off; head h occupies
// [base_off + h*head_dim, base_off + (h+1)*head_dim). Used for both the fused QKV
// buffer (token_stride=qkv_out_dim, base_off=0 for Q / q_dim for K) and separate Q/K
// buffers. Applied AFTER the QKV projection and BEFORE RoPE (Qwen3 has no QKV bias).
__global__ void qk_norm_kernel(
    half * __restrict__ data, const half * __restrict__ weight,
    int head_dim, int token_stride, int base_off, float eps)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    half * v = data + (int64_t)b * token_stride + base_off + (int64_t)h * head_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float x = __half2float(v[i]);
        sum_sq += x * x;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane < blockDim.x / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ float s_inv;
    if (threadIdx.x == 0) s_inv = rsqrtf(sum_sq / (float)head_dim + eps);
    __syncthreads();
    float inv = s_inv;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float x = __half2float(v[i]) * inv * __half2float(weight[i]);
        v[i] = __float2half(x);
    }
}

extern "C" void launch_qk_norm(
    half * data, const half * weight, int n_heads, int head_dim,
    int token_stride, int base_off, int batch, float eps, cudaStream_t stream)
{
    int threads = (head_dim < 256) ? head_dim : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    dim3 grid(batch, n_heads);
    qk_norm_kernel<<<grid, threads, 0, stream>>>(data, weight, head_dim, token_stride, base_off, eps);
}

// DeltaNet / gated-delta-net (Qwen3.5 linear-attention)

// Gates: beta = sigmoid(beta_raw); g = softplus(alpha_raw + dt) * a.  In place on beta,g.
__global__ void la_gates_kernel(half * beta, half * g, const half * dt, const half * a, int Hv, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * Hv) return;
    int h = idx % Hv;
    float b = __half2float(beta[idx]);
    beta[idx] = __float2half(1.0f / (1.0f + expf(-b)));
    float al = __half2float(g[idx]) + __half2float(dt[h]);
    float sp = (al > 20.0f) ? al : log1pf(expf(al));   // softplus, numerically safe
    g[idx] = __float2half(sp * __half2float(a[h]));
}
extern "C" void launch_la_gates(half * beta, half * g, const half * dt, const half * a, int Hv, int batch, cudaStream_t stream) {
    int n = batch * Hv; int t = 256;
    la_gates_kernel<<<(n + t - 1) / t, t, 0, stream>>>(beta, g, dt, a, Hv, batch);
}

// Split the fused DeltaNet projection GEMM output (row stride = conv_dim+value_dim+2*Hv,
// column layout [qkv | z | beta | g]) into its 4 contiguous destination buffers.
__global__ void dn_proj_split_kernel(
    const half * fused, half * qkv, half * z, half * beta, half * g,
    int conv_dim, int value_dim, int Hv, int proj_n, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * proj_n) return;
    int b = idx / proj_n, c = idx % proj_n;
    half v = fused[(int64_t)b * proj_n + c];
    if (c < conv_dim)                          qkv [(int64_t)b * conv_dim  + c]                          = v;
    else if (c < conv_dim + value_dim)         z   [(int64_t)b * value_dim + (c - conv_dim)]             = v;
    else if (c < conv_dim + value_dim + Hv)    beta[(int64_t)b * Hv        + (c - conv_dim - value_dim)] = v;
    else                                       g   [(int64_t)b * Hv        + (c - conv_dim - value_dim - Hv)] = v;
}

extern "C" void launch_dn_proj_split(const half * fused, half * qkv, half * z, half * beta, half * g,
    int conv_dim, int value_dim, int Hv, int batch, cudaStream_t stream) {
    int proj_n = conv_dim + value_dim + Hv + Hv;
    int n = batch * proj_n, t = 256;
    dn_proj_split_kernel<<<(n + t - 1) / t, t, 0, stream>>>(fused, qkv, z, beta, g, conv_dim, value_dim, Hv, proj_n, batch);
}

// Causal depthwise conv1d over [conv_state || new_input] + SiLU, then shift/append conv state.
// DECODE/per-token: 1 new token per sequence. conv weight w[c*K+j]; state layout [K-1, conv_dim].
__global__ void la_conv1d_kernel(
    half * out, const half * in, half * state, const half * w,
    int K, int conv_dim, int conv_size_per_seq, const int32_t * seq_ids, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * conv_dim) return;
    int b = idx / conv_dim, c = idx % conv_dim;
    int sid = seq_ids[b];
    half * st = state + (int64_t)sid * conv_size_per_seq;   // st[j*conv_dim + c], j in [0,K-1)
    float acc = 0.0f;
    for (int j = 0; j < K - 1; j++) acc += __half2float(w[c * K + j]) * __half2float(st[j * conv_dim + c]);
    float xin = __half2float(in[(int64_t)b * conv_dim + c]);
    acc += __half2float(w[c * K + (K - 1)]) * xin;
    float o = acc / (1.0f + expf(-acc));                    // SiLU
    out[(int64_t)b * conv_dim + c] = __float2half(o);
    for (int j = 0; j < K - 2; j++) st[j * conv_dim + c] = st[(j + 1) * conv_dim + c];  // shift
    st[(K - 2) * conv_dim + c] = __float2half(xin);                                      // append
}
extern "C" void launch_la_conv1d(half * out, const half * in, half * state, const half * w,
    int K, int conv_dim, int conv_size_per_seq, const int32_t * seq_ids, int batch, cudaStream_t stream) {
    int n = batch * conv_dim; int t = 256;
    la_conv1d_kernel<<<(n + t - 1) / t, t, 0, stream>>>(out, in, state, w, K, conv_dim, conv_size_per_seq, seq_ids, batch);
}

// Per-head L2 normalize (unit norm) of a [head_dim] slice, in place. grid=(batch,n_heads).
__global__ void l2_norm_kernel(half * data, int head_dim, int token_stride, int base_off, float eps) {
    int b = blockIdx.x, h = blockIdx.y;
    half * v = data + (int64_t)b * token_stride + base_off + (int64_t)h * head_dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) { float x = __half2float(v[i]); ss += x * x; }
    for (int o = warpSize / 2; o > 0; o /= 2) ss += __shfl_down_sync(0xffffffff, ss, o);
    __shared__ float sh[32]; int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    if (lane == 0) sh[wid] = ss; __syncthreads();
    if (wid == 0) { ss = (lane < blockDim.x / warpSize) ? sh[lane] : 0.0f;
        for (int o = warpSize / 2; o > 0; o /= 2) ss += __shfl_down_sync(0xffffffff, ss, o); }
    __shared__ float s_inv; if (threadIdx.x == 0) s_inv = rsqrtf(ss + eps); __syncthreads();
    float inv = s_inv;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) v[i] = __float2half(__half2float(v[i]) * inv);
}
extern "C" void launch_l2_norm(half * data, int head_dim, int n_heads, int token_stride, int base_off, int batch, float eps, cudaStream_t stream) {
    int t = (head_dim < 256) ? head_dim : 256; t = ((t + 31) / 32) * 32; if (t < 32) t = 32;
    dim3 grid(batch, n_heads);
    l2_norm_kernel<<<grid, t, 0, stream>>>(data, head_dim, token_stride, base_off, eps);
}

// Gated delta rule (decode, 1 token/seq). State M[j][i] stored at ssm[j*S+i] (row j contiguous).
// grid=(batch,H_v); blockDim.x=S; thread j owns row j. conv buffer layout per token:
// [ q(key_dim) | k(key_dim) | v(value_dim) ]; q/k indexed by k-head hk=h/(Hv/Hk), v by h.
__global__ void delta_rule_kernel(
    half * o, const half * conv, const half * g, const half * beta,
    half * ssm, const int32_t * seq_ids,
    int S, int Hk, int Hv, int key_dim, int value_dim, int ssm_size_per_seq, int conv_dim, int batch)
{
    int b = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    if (b >= batch || h >= Hv || j >= S) return;
    int sid = seq_ids[b];
    // GQA value->key head map: the reference expands q/k from Hk to Hv via ggml_repeat, which TILES
    // (value head h uses key head h % Hk: 0,1,..,Hk-1,0,1,..), NOT block-grouping (h/(Hv/Hk)). These
    // agree only when Hv==Hk (e.g. 0.8B) - for Hv!=Hk (4B: Hv=32,Hk=16) block-grouping reads the wrong
    // shared q/k for most value heads -> garbled output. Match the reference: h % Hk.
    int hk = h % Hk;
    const half * cbase = conv + (int64_t)b * conv_dim;
    __shared__ float sh_q[256], sh_k[256];
    sh_q[j] = __half2float(cbase[hk * S + j]);
    sh_k[j] = __half2float(cbase[key_dim + hk * S + j]);
    __syncthreads();
    float vj = __half2float(cbase[2 * key_dim + h * S + j]);
    float gh = expf(__half2float(g[b * Hv + h]));
    float bh = __half2float(beta[b * Hv + h]);
    float scale = rsqrtf((float)S);
    // State M[j][i] stored TRANSPOSED as Mbase[i*S + j] (column-major) so that at each loop
    // iteration i the warp (threads j=0..31) accesses S contiguous floats -> fully coalesced.
    // (Row-major Mbase[j*S+i] would make threads read stride-S apart, wasting DRAM bandwidth;
    // delta_rule is the DeltaNet decode bottleneck, memory-bound on this SxS state.)
    half * Mbase = ssm + (int64_t)sid * ssm_size_per_seq + (int64_t)h * S * S;
    // State stored in f16 (halves DRAM traffic; delta_rule is memory-bound). Compute stays f32.
    // Two passes (dj needs full Mk first) but only one write pass: pass1 reads M, accumulates
    // Mk = sum M[j][i]*gh*k[i]; pass2 re-reads M, writes decayed+updated state once (M*gh + k*delta).
    float Mk = 0.0f;
    for (int i = 0; i < S; i++) { Mk += __half2float(Mbase[i * S + j]) * gh * sh_k[i]; }
    float dj = bh * (vj - Mk);
    float oj = 0.0f;
    for (int i = 0; i < S; i++) { int idx = i * S + j; float m = __half2float(Mbase[idx]) * gh + sh_k[i] * dj; Mbase[idx] = __float2half(m); oj += m * sh_q[i]; }
    o[(int64_t)b * value_dim + h * S + j] = __float2half(oj * scale);
}
extern "C" void launch_delta_rule(half * o, const half * conv, const half * g, const half * beta,
    half * ssm, const int32_t * seq_ids, int S, int Hk, int Hv, int key_dim, int value_dim,
    int ssm_size_per_seq, int conv_dim, int batch, cudaStream_t stream) {
    dim3 grid(batch, Hv);
    delta_rule_kernel<<<grid, S, 0, stream>>>(o, conv, g, beta, ssm, seq_ids, S, Hk, Hv, key_dim, value_dim, ssm_size_per_seq, conv_dim, batch);
}

// Fused delta rule (mirrors vLLM's single-kernel recurrence): folds the per-head L2-norm of q,k
// and the gate computation (beta=sigmoid(beta_raw), g=softplus(alpha_raw+dt)*a) into this kernel,
// so the separate launch_la_gates + 2xlaunch_l2_norm launches disappear and q/k/g/beta no
// longer round-trip DRAM between kernels. Inputs are raw: conv=silu'd q|k|v (not normed), g_raw =
// ssm_alpha projection, beta_raw = ssm_beta projection; dt,a are the [Hv] gate weights. Requires
// S a multiple of 32 (S=128). blockDim.x = S, one thread per state row j; grid=(batch,Hv).
__global__ void delta_rule_fused_kernel(
    half * o, const half * conv, const half * g_raw, const half * beta_raw,
    const half * dt, const half * a,
    half * ssm, const int32_t * seq_ids,
    int S, int Hk, int Hv, int key_dim, int value_dim, int ssm_size_per_seq, int conv_dim,
    float eps, int batch, int cache)
{
    int b = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    if (b >= batch || h >= Hv || j >= S) return;
    int sid = seq_ids[b];
    // GQA value->key head map: the reference expands q/k from Hk to Hv via ggml_repeat, which TILES
    // (value head h uses key head h % Hk: 0,1,..,Hk-1,0,1,..), NOT block-grouping (h/(Hv/Hk)). These
    // agree only when Hv==Hk (e.g. 0.8B) - for Hv!=Hk (4B: Hv=32,Hk=16) block-grouping reads the wrong
    // shared q/k for most value heads -> garbled output. Match the reference: h % Hk.
    int hk = h % Hk;
    const half * cbase = conv + (int64_t)b * conv_dim;
    float qj = __half2float(cbase[hk * S + j]);
    float kj = __half2float(cbase[key_dim + hk * S + j]);

    // Fused per-head L2-norm of q and k (replaces two launch_l2_norm kernels)
    // Block reduction of Σq^2 and Σk^2 over the S rows (one k-head's head_dim). Normalize in
    // shared memory only - the global conv buffer is never rewritten (delta_rule is its only reader).
    int lane = j & 31, wid = j >> 5, nwarp = (S + 31) >> 5;
    float qq = qj * qj, kk = kj * kj;
    for (int off = 16; off > 0; off >>= 1) { qq += __shfl_down_sync(0xffffffff, qq, off); kk += __shfl_down_sync(0xffffffff, kk, off); }
    __shared__ float wq[8], wk[8];
    if (lane == 0) { wq[wid] = qq; wk[wid] = kk; }
    __syncthreads();
    if (wid == 0) {
        qq = (lane < nwarp) ? wq[lane] : 0.0f; kk = (lane < nwarp) ? wk[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) { qq += __shfl_down_sync(0xffffffff, qq, off); kk += __shfl_down_sync(0xffffffff, kk, off); }
        if (lane == 0) { wq[0] = qq; wk[0] = kk; }
    }
    __syncthreads();
    float qinv = rsqrtf(wq[0] + eps), kinv = rsqrtf(wk[0] + eps);
    __shared__ float sh_q[256], sh_k[256];
    sh_q[j] = qj * qinv;
    sh_k[j] = kj * kinv;
    __syncthreads();

    float vj = __half2float(cbase[2 * key_dim + h * S + j]);
    // Fused gates (replaces launch_la_gates)
    float bh = 1.0f / (1.0f + expf(-__half2float(beta_raw[b * Hv + h])));       // sigmoid(beta)
    float al = __half2float(g_raw[b * Hv + h]) + __half2float(dt[h]);
    float sp = (al > 20.0f) ? al : log1pf(expf(al));                            // softplus
    float gh = expf(sp * __half2float(a[h]));                                   // exp(g)
    float scale = rsqrtf((float)S);

    // State M[j][i] stored TRANSPOSED as Mbase[i*S+j] (coalesced), f16 storage / f32 compute.
    // The state is read twice (Mk reduction, then update); at ~3 FMAs / 2-byte load this is
    // memory-bound. `cache` (ALPHA_DN_CACHE, launched with S*S*2 dynamic shared) caches thread j's
    // OWN state column in shared during pass 1 so pass 2 skips the 2nd global read -> ~1/3 less state
    // traffic. Each thread only touches its own column, so no __syncthreads is needed. Math identical.
    half * Mbase = ssm + (int64_t)sid * ssm_size_per_seq + (int64_t)h * S * S;
    extern __shared__ half sh_state[];   // [S*S] dynamic; only used when cache != 0
    float Mk = 0.0f, dj, oj = 0.0f;
    if (cache) {
        for (int i = 0; i < S; i++) { half m = Mbase[i * S + j]; sh_state[i * S + j] = m; Mk += __half2float(m) * gh * sh_k[i]; }
        dj = bh * (vj - Mk);
        for (int i = 0; i < S; i++) { int idx = i * S + j; float m = __half2float(sh_state[idx]) * gh + sh_k[i] * dj; Mbase[idx] = __float2half(m); oj += m * sh_q[i]; }
    } else {
        for (int i = 0; i < S; i++) { Mk += __half2float(Mbase[i * S + j]) * gh * sh_k[i]; }
        dj = bh * (vj - Mk);
        for (int i = 0; i < S; i++) { int idx = i * S + j; float m = __half2float(Mbase[idx]) * gh + sh_k[i] * dj; Mbase[idx] = __float2half(m); oj += m * sh_q[i]; }
    }
    float oj_final = oj * scale;
    int64_t oidx = (int64_t)b * value_dim + h * S + j;
    o[oidx] = __float2half(oj_final);
}
extern "C" void launch_delta_rule_fused(half * o, const half * conv, const half * g_raw, const half * beta_raw,
    const half * dt, const half * a, half * ssm, const int32_t * seq_ids, int S, int Hk, int Hv,
    int key_dim, int value_dim, int ssm_size_per_seq, int conv_dim, float eps, int batch,
    cudaStream_t stream) {
    dim3 grid(batch, Hv);
    // ALPHA_DN_CACHE: cache the per-thread state column in shared to drop the 2nd global state read.
    // Only when the S*S f16 block fits shared (S<=128 -> 32 KiB, under the 48 KiB default). Trades
    // occupancy (32 KiB/block -> ~1 block/SM) for bandwidth.
    static const int dn_cache = getenv("ALPHA_DN_CACHE") ? atoi(getenv("ALPHA_DN_CACHE")) : 0;
    size_t sh_bytes = (size_t)S * S * sizeof(half);
    int use_cache = (dn_cache && sh_bytes <= 46u * 1024) ? 1 : 0;
    delta_rule_fused_kernel<<<grid, S, use_cache ? sh_bytes : 0, stream>>>(o, conv, g_raw, beta_raw, dt, a, ssm, seq_ids,
        S, Hk, Hv, key_dim, value_dim, ssm_size_per_seq, conv_dim, eps, batch, use_cache);
}

// Gated RMSNorm: y = RMSNorm(o; weight[S]) * SiLU(z), per head. grid=(batch,H_v).
__global__ void gated_rmsnorm_kernel(half * y, const half * o, const half * z, const half * w, int S, int value_dim, float eps) {
    int b = blockIdx.x, h = blockIdx.y;
    const half * oh = o + (int64_t)b * value_dim + (int64_t)h * S;
    const half * zh = z + (int64_t)b * value_dim + (int64_t)h * S;
    half * yh = y + (int64_t)b * value_dim + (int64_t)h * S;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) { float x = __half2float(oh[i]); ss += x * x; }
    for (int off = warpSize / 2; off > 0; off /= 2) ss += __shfl_down_sync(0xffffffff, ss, off);
    __shared__ float sh[32]; int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    if (lane == 0) sh[wid] = ss; __syncthreads();
    if (wid == 0) { ss = (lane < blockDim.x / warpSize) ? sh[lane] : 0.0f;
        for (int off = warpSize / 2; off > 0; off /= 2) ss += __shfl_down_sync(0xffffffff, ss, off); }
    __shared__ float s_inv; if (threadIdx.x == 0) s_inv = rsqrtf(ss / (float)S + eps); __syncthreads();
    float inv = s_inv;
    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        float on = __half2float(oh[i]) * inv * __half2float(w[i]);
        float zz = __half2float(zh[i]); float sz = zz / (1.0f + expf(-zz));
        yh[i] = __float2half(on * sz);
    }
}
extern "C" void launch_gated_rmsnorm(half * y, const half * o, const half * z, const half * w, int S, int Hv, int value_dim, int batch, float eps, cudaStream_t stream) {
    int t = (S < 256) ? S : 256; t = ((t + 31) / 32) * 32; if (t < 32) t = 32;
    dim3 grid(batch, Hv);
    gated_rmsnorm_kernel<<<grid, t, 0, stream>>>(y, o, z, w, S, value_dim, eps);
}

// Qwen3.5/Qwen3Next gated attention kernels

// Split interleaved [query(head_dim) | gate(head_dim)] per head into contiguous
// query[n_head*head_dim] and gate[n_head*head_dim]. qg layout per token: for head h,
// [h*2*head_dim .. +head_dim) = query, [.. +head_dim .. +2*head_dim) = gate.
__global__ void deinterleave_qgate_kernel(half * q_out, half * gate_out, const half * qg,
                                          int head_dim, int n_head, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int QD = n_head * head_dim;
    if (idx >= batch * QD) return;
    int b = idx / QD, rem = idx % QD, h = rem / head_dim, d = rem % head_dim;
    const half * src = qg + (int64_t)b * 2 * QD + (int64_t)h * 2 * head_dim;
    q_out[(int64_t)b * QD + h * head_dim + d]    = src[d];
    gate_out[(int64_t)b * QD + h * head_dim + d] = src[head_dim + d];
}
extern "C" void launch_deinterleave_qgate(half * q_out, half * gate_out, const half * qg,
                                          int head_dim, int n_head, int batch, cudaStream_t stream) {
    int n = batch * n_head * head_dim, t = 256;
    deinterleave_qgate_kernel<<<(n + t - 1) / t, t, 0, stream>>>(q_out, gate_out, qg, head_dim, n_head, batch);
}

// Partial RoPE: rotate only the first n_rot dims of each head; dims [n_rot,head_dim) untouched.
// Split-half pairing (j, j+n_rot/2) [ggml NEOX].
// theta normalized by n_rot (llama.cpp partial-rope convention).
__global__ void partial_rope_kernel(half * q, half * k, const int32_t * positions,
    int head_dim, int n_rot, int n_head_q, int n_head_k, float freq_base, float freq_scale) {
    int b = blockIdx.x; int pos = positions[b];
    int half_rot = n_rot / 2;
    half * qr = q + (int64_t)b * n_head_q * head_dim;
    half * kr = k + (int64_t)b * n_head_k * head_dim;
    for (int idx = threadIdx.x; idx < n_head_q * half_rot; idx += blockDim.x) {
        int h = idx / half_rot, j = idx % half_rot;
        float theta = (float)pos / (powf(freq_base, 2.0f * j / (float)n_rot) * freq_scale);
        float c = cosf(theta), s = sinf(theta);
        half * base = qr + (int64_t)h * head_dim;
        int i0 = j, i1 = j + half_rot;
        float x0 = __half2float(base[i0]), x1 = __half2float(base[i1]);
        base[i0] = __float2half(x0 * c - x1 * s);
        base[i1] = __float2half(x0 * s + x1 * c);
    }
    for (int idx = threadIdx.x; idx < n_head_k * half_rot; idx += blockDim.x) {
        int h = idx / half_rot, j = idx % half_rot;
        float theta = (float)pos / (powf(freq_base, 2.0f * j / (float)n_rot) * freq_scale);
        float c = cosf(theta), s = sinf(theta);
        half * base = kr + (int64_t)h * head_dim;
        int i0 = j, i1 = j + half_rot;
        float x0 = __half2float(base[i0]), x1 = __half2float(base[i1]);
        base[i0] = __float2half(x0 * c - x1 * s);
        base[i1] = __float2half(x0 * s + x1 * c);
    }
}
extern "C" void launch_partial_rope(half * q, half * k, const int32_t * positions,
    int head_dim, int n_rot, int n_head_q, int n_head_k, int batch,
    float freq_base, float freq_scale, cudaStream_t stream) {
    int max_pairs = n_head_q * (n_rot / 2);
    int t = (max_pairs < 256) ? max_pairs : 256; t = ((t + 31) / 32) * 32; if (t < 32) t = 32;
    partial_rope_kernel<<<batch, t, 0, stream>>>(q, k, positions, head_dim, n_rot, n_head_q, n_head_k, freq_base, freq_scale);
}

// Output gate: attn_out[i] *= sigmoid(gate[i]), over [batch, n_head*head_dim].
__global__ void gate_sigmoid_mul_kernel(half * attn_out, const half * gate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = __half2float(gate[idx]);
    attn_out[idx] = __float2half(__half2float(attn_out[idx]) / (1.0f + expf(-g)));
}
extern "C" void launch_gate_sigmoid_mul(half * attn_out, const half * gate, int n, int batch, cudaStream_t stream) {
    int total = n * batch, t = 256;
    gate_sigmoid_mul_kernel<<<(total + t - 1) / t, t, 0, stream>>>(attn_out, gate, total);
}

// RoPE (Rotary Position Embedding)
// Apply RoPE to Q and K tensors in-place
// Layout: [batch, n_heads, head_dim] where pairs (2i, 2i+1) are rotated

__global__ void rope_kernel(
    half * __restrict__ q,
    half * __restrict__ k,
    const int32_t * __restrict__ positions,
    int head_dim,
    int n_head_q,
    int n_head_k,
    float freq_base,
    float freq_scale)
{
    const int batch_idx = blockIdx.x;
    const int pos = positions[batch_idx];

    const int total_q = n_head_q * head_dim;
    const int total_k = n_head_k * head_dim;

    half * q_row = q + batch_idx * total_q;
    half * k_row = k + batch_idx * total_k;

    // NEOX (split-half) pairing (j, j+head_dim/2) - matches llama.cpp NEOX rope (LLaMA/Qwen).
    const int hd2 = head_dim / 2;
    // Process Q heads
    for (int idx = threadIdx.x; idx < total_q / 2; idx += blockDim.x) {
        int dim_pair = idx % hd2;
        float theta = (float)pos / (powf(freq_base, 2.0f * dim_pair / (float)head_dim) * freq_scale);
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        int hbase = (idx / hd2) * head_dim;
        int i0 = hbase + dim_pair, i1 = hbase + dim_pair + hd2;
        float x0 = __half2float(q_row[i0]);
        float x1 = __half2float(q_row[i1]);
        q_row[i0] = __float2half(x0 * cos_t - x1 * sin_t);
        q_row[i1] = __float2half(x0 * sin_t + x1 * cos_t);
    }

    // Process K heads
    for (int idx = threadIdx.x; idx < total_k / 2; idx += blockDim.x) {
        int dim_pair = idx % hd2;
        float theta = (float)pos / (powf(freq_base, 2.0f * dim_pair / (float)head_dim) * freq_scale);
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        int hbase = (idx / hd2) * head_dim;
        int i0 = hbase + dim_pair, i1 = hbase + dim_pair + hd2;
        float x0 = __half2float(k_row[i0]);
        float x1 = __half2float(k_row[i1]);
        k_row[i0] = __float2half(x0 * cos_t - x1 * sin_t);
        k_row[i1] = __float2half(x0 * sin_t + x1 * cos_t);
    }
}

extern "C" void launch_rope(
    half * q, half * k, const int32_t * positions,
    int head_dim, int n_head_q, int n_head_k, int batch,
    float freq_base, float freq_scale, cudaStream_t stream)
{
    int max_pairs = (n_head_q * head_dim) / 2;
    int threads = (max_pairs < 256) ? max_pairs : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    rope_kernel<<<batch, threads, 0, stream>>>(
        q, k, positions, head_dim, n_head_q, n_head_k, freq_base, freq_scale);
}

// SiLU x Multiply
// out[i] = SiLU(gate[i]) * up[i]
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

__global__ void silu_mul_kernel(
    half * __restrict__ out,
    const half * __restrict__ gate,
    const half * __restrict__ up,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu = g / (1.0f + expf(-g));
    out[idx] = __float2half(silu * u);
}

extern "C" void launch_silu_mul(
    half * out, const half * gate, const half * up,
    int n, int batch, cudaStream_t stream)
{
    int total = n * batch;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(out, gate, up, total);
}

// Token Embedding Lookup
// out[b] = embedding_table[token_ids[b]]
// embedding_table: [vocab_size, n_embd] in FP16

__global__ void embed_tokens_kernel(
    half * __restrict__ out,
    const half * __restrict__ table,
    const int32_t * __restrict__ token_ids,
    int n_embd)
{
    int batch_idx = blockIdx.x;
    int tok = token_ids[batch_idx];

    const half * src = table + (int64_t)tok * n_embd;
    half * dst = out + (int64_t)batch_idx * n_embd;

    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        dst[i] = src[i];
    }
}

extern "C" void launch_embed_tokens(
    half * out, const half * table, const int32_t * token_ids,
    int n_embd, int batch, cudaStream_t stream)
{
    int threads = (n_embd < 256) ? n_embd : 256;
    threads = ((threads + 31) / 32) * 32;
    embed_tokens_kernel<<<batch, threads, 0, stream>>>(out, table, token_ids, n_embd);
}

// Residual Addition
// out[i] += residual[i]

__global__ void add_residual_kernel(
    half * __restrict__ out,
    const half * __restrict__ residual,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float a = __half2float(out[idx]);
    float b = __half2float(residual[idx]);
    out[idx] = __float2half(a + b);
}

extern "C" void launch_add_residual(
    half * out, const half * residual,
    int n, int batch, cudaStream_t stream)
{
    int total = n * batch;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_residual_kernel<<<blocks, threads, 0, stream>>>(out, residual, total);
}

// Fused Deinterleave QKV + RoPE
// Combines split(QKV) + RoPE(Q,K) into one kernel launch.
// Reads fused QKV matmul output [batch, q_dim+kv_dim+kv_dim],
// writes Q/K with RoPE applied, copies V directly.
// Replaces: launch_deinterleave_qkv + launch_rope (2 launches -> 1).

__global__ void fused_deinterleave_rope_kernel(
    half * __restrict__ q_out,
    half * __restrict__ k_out,
    half * __restrict__ v_out,
    const half * __restrict__ qkv,
    const int32_t * __restrict__ positions,
    const half * __restrict__ bq,
    const half * __restrict__ bk,
    const half * __restrict__ bv,
    int q_dim, int kv_dim, int qkv_dim,
    int head_dim,
    float freq_base, float freq_scale)
{
    const int b = blockIdx.x;
    const int pos = positions[b];

    const half * src = qkv + (int64_t)b * qkv_dim;
    half * dq = q_out + (int64_t)b * q_dim;
    half * dk = k_out + (int64_t)b * kv_dim;
    half * dv = v_out + (int64_t)b * kv_dim;

    // Q: split from fused buffer + add bias + apply RoPE.
    // NEOX (split-half) pairing: rotate (j, j+head_dim/2), matching llama.cpp NEOX rope used by
    // LLaMA/Qwen (adjacent (2j,2j+1) pairing is GPT-J style and garbles NEOX models).
    const int hd2 = head_dim / 2;
    for (int idx = threadIdx.x; idx < q_dim / 2; idx += blockDim.x) {
        int dim_pair = idx % hd2;
        int hbase = (idx / hd2) * head_dim;
        int i0 = hbase + dim_pair, i1 = hbase + dim_pair + hd2;

        float theta = (float)pos / (powf(freq_base, 2.0f * dim_pair / (float)head_dim) * freq_scale);
        float cos_t = cosf(theta), sin_t = sinf(theta);

        float x0 = __half2float(src[i0]);
        float x1 = __half2float(src[i1]);
        if (bq) { x0 = __half2float(__float2half(x0 + __half2float(bq[i0])));
                   x1 = __half2float(__float2half(x1 + __half2float(bq[i1]))); }
        dq[i0] = __float2half(x0 * cos_t - x1 * sin_t);
        dq[i1] = __float2half(x0 * sin_t + x1 * cos_t);
    }

    // K: split from fused buffer + add bias + apply RoPE (NEOX split-half pairing)
    for (int idx = threadIdx.x; idx < kv_dim / 2; idx += blockDim.x) {
        int dim_pair = idx % hd2;
        int hbase = (idx / hd2) * head_dim;
        int i0 = hbase + dim_pair, i1 = hbase + dim_pair + hd2;

        float theta = (float)pos / (powf(freq_base, 2.0f * dim_pair / (float)head_dim) * freq_scale);
        float cos_t = cosf(theta), sin_t = sinf(theta);

        float x0 = __half2float(src[q_dim + i0]);
        float x1 = __half2float(src[q_dim + i1]);
        if (bk) { x0 = __half2float(__float2half(x0 + __half2float(bk[i0])));
                   x1 = __half2float(__float2half(x1 + __half2float(bk[i1]))); }
        dk[i0] = __float2half(x0 * cos_t - x1 * sin_t);
        dk[i1] = __float2half(x0 * sin_t + x1 * cos_t);
    }

    // V: copy + add bias (no RoPE)
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        if (bv) dv[i] = __hadd(src[q_dim + kv_dim + i], bv[i]);
        else    dv[i] = src[q_dim + kv_dim + i];
    }
}

extern "C" void launch_fused_deinterleave_rope(
    half * q, half * k, half * v, const half * qkv,
    const int32_t * positions,
    const half * bq, const half * bk, const half * bv,
    int q_dim, int kv_dim, int head_dim, int batch_size,
    float freq_base, float freq_scale, cudaStream_t stream)
{
    int qkv_dim = q_dim + 2 * kv_dim;
    int max_pairs = q_dim / 2;
    int threads = (max_pairs < 256) ? max_pairs : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    fused_deinterleave_rope_kernel<<<batch_size, threads, 0, stream>>>(
        q, k, v, qkv, positions, bq, bk, bv,
        q_dim, kv_dim, qkv_dim, head_dim,
        freq_base, freq_scale);
}

// Fused SiLU x Multiply (from interleaved gate+up buffer)
// Reads gate+up from a single fused matmul output [batch, 2*n_ff],
// applies SiLU(gate) * up in one pass.
// Replaces: launch_deinterleave_qkv(gate,up) + launch_silu_mul (2 launches -> 1).

__global__ void silu_mul_interleaved_kernel(
    half * __restrict__ out,
    const half * __restrict__ gate_up,
    int n_ff, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int b = idx / n_ff;
    int i = idx % n_ff;

    const half * src = gate_up + (int64_t)b * 2 * n_ff;
    float g = __half2float(src[i]);
    float u = __half2float(src[n_ff + i]);
    float silu = g / (1.0f + expf(-g));
    out[idx] = __float2half(silu * u);
}

extern "C" void launch_silu_mul_interleaved(
    half * out, const half * gate_up,
    int n_ff, int batch, cudaStream_t stream)
{
    int total = n_ff * batch;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    silu_mul_interleaved_kernel<<<blocks, threads, 0, stream>>>(out, gate_up, n_ff, total);
}

// Causal Self-Attention for Prefill
// Simple QK^T with causal mask + softmax + attnxV for short sequences (N <= 512).
// Each block handles one query head. Threads cooperate on the NxN score matrix.
// GQA: multiple query heads share one KV head (h_kv = h_q / gqa_ratio).
// Layout: Q[N, n_head, head_dim], K/V[N, n_kv_head, head_dim] (interleaved per-token)

__global__ void causal_attn_prefill_kernel(
    half * __restrict__ output,      // [N, n_head, head_dim]
    const half * __restrict__ Q,     // [N, n_head, head_dim]
    const half * __restrict__ K,     // [N, n_kv_head, head_dim]
    const half * __restrict__ V,     // [N, n_kv_head, head_dim]
    int N, int n_head, int n_kv_head, int head_dim, float sm_scale,
    int window)                      // sliding-window width (>0 -> attend only to last `window`; <=0 -> full causal)
{
    int h = blockIdx.x;              // query head
    int h_kv = h / (n_head / n_kv_head);  // KV head (GQA)

    // Each thread handles one query position
    int q_pos = threadIdx.x;
    if (q_pos >= N) return;

    int q_stride = n_head * head_dim;
    int kv_stride = n_kv_head * head_dim;

    // Compute QK^T scores for this query position. Full causal attends 0..q_pos; sliding-window
    // attends [k_lo, q_pos] with k_lo = q_pos-window+1 (clamped >=0).
    int k_lo = (window > 0 && q_pos - window + 1 > 0) ? (q_pos - window + 1) : 0;
    extern __shared__ float shared_mem[];
    float * scores = shared_mem + q_pos * N;  // each thread gets N floats

    float max_score = -1e30f;
    for (int k_pos = k_lo; k_pos <= q_pos; k_pos++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = __half2float(Q[q_pos * q_stride + h * head_dim + d]);
            float k_val = __half2float(K[k_pos * kv_stride + h_kv * head_dim + d]);
            dot += q_val * k_val;
        }
        dot *= sm_scale;
        scores[k_pos] = dot;
        if (dot > max_score) max_score = dot;
    }

    // Softmax
    float sum_exp = 0.0f;
    for (int k_pos = k_lo; k_pos <= q_pos; k_pos++) {
        scores[k_pos] = expf(scores[k_pos] - max_score);
        sum_exp += scores[k_pos];
    }
    float inv_sum = 1.0f / (sum_exp + 1e-8f);

    // Attn x V: weighted sum of V vectors
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int v_pos = k_lo; v_pos <= q_pos; v_pos++) {
            float v_val = __half2float(V[v_pos * kv_stride + h_kv * head_dim + d]);
            acc += scores[v_pos] * inv_sum * v_val;
        }
        output[q_pos * q_stride + h * head_dim + d] = __float2half(acc);
    }
}

extern "C" void launch_causal_attn_prefill(
    half * output, const half * Q, const half * K, const half * V,
    int N, int n_head, int n_kv_head, int head_dim, float sm_scale,
    int window, cudaStream_t stream)
{
    // One block per query head, N threads per block
    // Shared memory: N * N floats for the score matrix
    int threads = N;
    size_t smem = (size_t)N * N * sizeof(float);
    causal_attn_prefill_kernel<<<n_head, threads, smem, stream>>>(
        output, Q, K, V, N, n_head, n_kv_head, head_dim, sm_scale, window);
}

// Extract Last Token
// Copy the last token's hidden state from [N, dim] to [1, dim]
__global__ void extract_last_token_kernel(
    half * __restrict__ out, const half * __restrict__ in,
    int last_idx, int dim)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    out[d] = in[(int64_t)last_idx * dim + d];
}

extern "C" void launch_extract_last_token(
    half * out, const half * in, int last_idx, int dim, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    extract_last_token_kernel<<<blocks, threads, 0, stream>>>(out, in, last_idx, dim);
}

// Fused RMSNorm + Residual Save
// Combines: cudaMemcpyAsync(residual, hidden, D2D) + rms_norm(hidden, hidden, weight)
// into one kernel: residual = hidden; hidden = norm(hidden) * weight
// Saves one memory read of hidden_buf and eliminates the D2D memcpy sync point.

__global__ void fused_rmsnorm_residual_kernel(
    half * __restrict__ out,       // normalized output (can be same as inp)
    half * __restrict__ residual,  // receives copy of input before normalization
    const half * __restrict__ inp, // input (read once)
    const half * __restrict__ weight,
    int n, float eps)
{
    const int row = blockIdx.x;
    const half * x = inp + row * n;
    half * y = out + row * n;
    half * r = residual + row * n;

    // Pass 1: read input, save to residual, compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        half val = x[i];
        r[i] = val;  // save residual
        float v = __half2float(val);
        sum_sq += v * v;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    // Cross-warp reduction
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane < blockDim.x / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) s_rms_inv = rsqrtf(sum_sq / (float)n + eps);
    __syncthreads();
    float rms_inv = s_rms_inv;

    // Pass 2: normalize (read from residual to avoid re-reading inp when out==inp)
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __half2float(r[i]) * rms_inv;
        y[i] = __float2half(v * __half2float(weight[i]));
    }
}

extern "C" void launch_fused_rmsnorm_residual(
    half * out, half * residual, const half * inp, const half * weight,
    int n, int batch, float eps, cudaStream_t stream)
{
    int threads = (n < 1024) ? n : 1024;
    threads = ((threads + 31) / 32) * 32;
    fused_rmsnorm_residual_kernel<<<batch, threads, 0, stream>>>(
        out, residual, inp, weight, n, eps);
}

// Fused Add Residual + RMSNorm
// Combines: add_residual(hidden, residual) + cudaMemcpyAsync(residual, hidden) + rms_norm(hidden, hidden, weight)
// into one kernel: tmp = hidden + residual; residual = tmp; hidden = norm(tmp) * weight
// Eliminates 2 kernel launches + 1 D2D memcpy per occurrence.

__global__ void fused_add_residual_rmsnorm_kernel(
    half * __restrict__ out,       // normalized output
    half * __restrict__ residual,  // updated to (inp + old_residual), then used as norm input
    const half * __restrict__ inp, // current hidden state
    const half * __restrict__ weight,
    int n, float eps)
{
    const int row = blockIdx.x;
    const half * x = inp + row * n;
    half * y = out + row * n;
    half * r = residual + row * n;

    // Pass 1: add residual, save result, compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float sum = __half2float(x[i]) + __half2float(r[i]);
        half sum_h = __float2half(sum);
        r[i] = sum_h;  // residual = inp + residual
        sum_sq += sum * sum;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    // Cross-warp reduction
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane < blockDim.x / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) s_rms_inv = rsqrtf(sum_sq / (float)n + eps);
    __syncthreads();
    float rms_inv = s_rms_inv;

    // Pass 2: normalize from saved residual
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __half2float(r[i]) * rms_inv;
        y[i] = __float2half(v * __half2float(weight[i]));
    }
}

extern "C" void launch_fused_add_residual_rmsnorm(
    half * out, half * residual, const half * inp, const half * weight,
    int n, int batch, float eps, cudaStream_t stream)
{
    int threads = (n < 1024) ? n : 1024;
    threads = ((threads + 31) / 32) * 32;
    fused_add_residual_rmsnorm_kernel<<<batch, threads, 0, stream>>>(
        out, residual, inp, weight, n, eps);
}

// Fused KV Scatter (K + V in one kernel)
// Combines two launch_scatter_kv calls into one kernel launch.

__global__ void scatter_kv_pair_kernel(
    half * __restrict__ k_pool, half * __restrict__ v_pool,
    const half * __restrict__ k_src, const half * __restrict__ v_src,
    const int64_t * __restrict__ offsets,
    int kv_dim, int batch_size)
{
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int64_t dst_offset = offsets[batch_idx];
    const half * ks = k_src + (int64_t)batch_idx * kv_dim;
    const half * vs = v_src + (int64_t)batch_idx * kv_dim;
    half * kd = k_pool + dst_offset;
    half * vd = v_pool + dst_offset;

    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        kd[i] = ks[i];
        vd[i] = vs[i];
    }
}

extern "C" void launch_scatter_kv_pair(
    half * k_pool, half * v_pool,
    const half * k_src, const half * v_src,
    const int64_t * offsets,
    int kv_dim, int batch_size, cudaStream_t stream)
{
    if (batch_size <= 0) return;
    int threads = (kv_dim < 256) ? kv_dim : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    scatter_kv_pair_kernel<<<batch_size, threads, 0, stream>>>(
        k_pool, v_pool, k_src, v_src, offsets, kv_dim, batch_size);
}

// MoE FFN: weighted scatter-add of expert outputs
// acc[idx[r], :] += w[r] * src[r, :]   (FP32 accumulator, FP16 expert output)
// Within one expert group token indices are unique, and groups run sequentially on the
// same stream -> plain (non-atomic) read-modify-write is race-free.

__global__ void moe_scatter_add_kernel(
    float * __restrict__ acc,
    const half * __restrict__ src,
    const int32_t * __restrict__ idx,
    const float * __restrict__ w,
    int n_embd)
{
    const int r = blockIdx.x;
    const int token = idx[r];
    const float wt = w[r];
    const half * s = src + (int64_t)r * n_embd;
    float * d = acc + (int64_t)token * n_embd;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        d[i] += wt * __half2float(s[i]);
    }
}

extern "C" void launch_moe_scatter_add(
    float * acc, const half * src, const int32_t * idx,
    const float * w, int n_embd, int count, cudaStream_t stream)
{
    if (count <= 0) return;
    int threads = (n_embd < 256) ? n_embd : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    moe_scatter_add_kernel<<<count, threads, 0, stream>>>(acc, src, idx, w, n_embd);
}

// MoE FFN: finalize
// out[b,i] = (half)( acc[b,i] + sigmoid(shexp_gate[b]) * shexp[b,i] )
// shexp == nullptr -> out = (half)acc. shexp_gate == nullptr -> shared expert weight 1.0.

__global__ void moe_finalize_kernel(
    half * __restrict__ out,
    const float * __restrict__ acc,
    const half * __restrict__ shexp,
    const half * __restrict__ gate,
    int n_embd)
{
    const int b = blockIdx.x;
    float g = 1.0f;
    if (shexp && gate) {
        float gl = __half2float(gate[b]);
        g = 1.0f / (1.0f + expf(-gl));
    }
    const float * a = acc + (int64_t)b * n_embd;
    half * o = out + (int64_t)b * n_embd;
    if (shexp) {
        const half * s = shexp + (int64_t)b * n_embd;
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
            o[i] = __float2half(a[i] + g * __half2float(s[i]));
        }
    } else {
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
            o[i] = __float2half(a[i]);
        }
    }
}

extern "C" void launch_moe_finalize(
    half * out, const float * acc, const half * shexp,
    const half * shexp_gate, int n_embd, int batch, cudaStream_t stream)
{
    if (batch <= 0) return;
    int threads = (n_embd < 256) ? n_embd : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    moe_finalize_kernel<<<batch, threads, 0, stream>>>(out, acc, shexp, shexp_gate, n_embd);
}
