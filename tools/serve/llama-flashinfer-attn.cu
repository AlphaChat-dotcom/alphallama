// FlashInfer contiguous decode attention kernel
// Called from ggml-cuda fattn.cu hook.
// Uses FlashInfer BatchDecode with PAGE_SIZE=16 page table over contiguous KV.
// All users processed in one kernel launch.

#ifdef ALPHA_HAS_FLASHINFER

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// FlashInfer headers
#include "../../vendor/flashinfer/include/flashinfer/attention/decode.cuh"
#include "../../vendor/flashinfer/include/flashinfer/attention/default_decode_params.cuh"
#include "../../vendor/flashinfer/include/flashinfer/attention/variants.cuh"

using namespace flashinfer;
using StdAttention = DefaultAttention<false, false, false, false>;

#define FI_PAGE_SIZE 16

// Pre-allocated GPU metadata
static struct {
    int32_t * d_indptr = nullptr;
    int32_t * d_indices = nullptr;
    int32_t * d_last_page_len = nullptr;
    int32_t * d_request_indices = nullptr;
    int32_t * d_kv_tile_indices = nullptr;
    int32_t * d_o_indptr = nullptr;
    int32_t * d_kv_chunk_size = nullptr;
    int alloc_batch = 0;
    int alloc_pages = 0;
} g_meta;

static void ensure_meta(int batch_size, int n_pages, cudaStream_t stream) {
    int pb = std::max(batch_size, 8);
    int pp = std::max(n_pages * batch_size, 256);
    if (g_meta.alloc_batch >= pb && g_meta.alloc_pages >= pp) return;

    if (g_meta.d_indptr) cudaFree(g_meta.d_indptr);
    if (g_meta.d_indices) cudaFree(g_meta.d_indices);
    if (g_meta.d_last_page_len) cudaFree(g_meta.d_last_page_len);
    if (g_meta.d_request_indices) cudaFree(g_meta.d_request_indices);
    if (g_meta.d_kv_tile_indices) cudaFree(g_meta.d_kv_tile_indices);
    if (g_meta.d_o_indptr) cudaFree(g_meta.d_o_indptr);
    if (g_meta.d_kv_chunk_size) cudaFree(g_meta.d_kv_chunk_size);

    g_meta.alloc_batch = pb;
    g_meta.alloc_pages = pp;

    cudaMalloc(&g_meta.d_indptr, (pb + 1) * sizeof(int32_t));
    cudaMalloc(&g_meta.d_indices, pp * sizeof(int32_t));
    cudaMalloc(&g_meta.d_last_page_len, pb * sizeof(int32_t));
    cudaMalloc(&g_meta.d_request_indices, pb * sizeof(int32_t));
    cudaMalloc(&g_meta.d_kv_tile_indices, pb * sizeof(int32_t));
    cudaMalloc(&g_meta.d_o_indptr, (pb + 1) * sizeof(int32_t));
    cudaMalloc(&g_meta.d_kv_chunk_size, sizeof(int32_t));

    // Static: request_indices = identity, kv_tile_indices = zeros, o_indptr = identity
    std::vector<int32_t> h_req(pb), h_zeros(pb, 0), h_oind(pb + 1);
    for (int i = 0; i < pb; i++) { h_req[i] = i; h_oind[i] = i; }
    h_oind[pb] = pb;
    cudaMemcpy(g_meta.d_request_indices, h_req.data(), pb * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(g_meta.d_kv_tile_indices, h_zeros.data(), pb * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(g_meta.d_o_indptr, h_oind.data(), (pb + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    int32_t chunk = 1 << 20;
    cudaMemcpy(g_meta.d_kv_chunk_size, &chunk, sizeof(int32_t), cudaMemcpyHostToDevice);
}

// BatchDecode with PAGE_SIZE=16 over contiguous KV
template <uint32_t HEAD_DIM>
static cudaError_t batch_decode_paged_impl(
    half * output, const half * q, half * k, half * v,
    int n_tokens, int kv_len, int n_head_q, int n_head_kv,
    float scale, cudaStream_t stream)
{
    // Map contiguous KV [kv_len, n_head_kv, HEAD_DIM] into pages of 16 tokens.
    int n_pages = (kv_len + FI_PAGE_SIZE - 1) / FI_PAGE_SIZE;
    int last_page_len = kv_len % FI_PAGE_SIZE;
    if (last_page_len == 0) last_page_len = FI_PAGE_SIZE;

    int pb = 1;
    while (pb < n_tokens) pb *= 2;
    pb = std::max(pb, 1);

    ensure_meta(pb, n_pages, stream);

    // Cache page table - only rebuild when n_pages or batch changes
    static int cached_n_pages = 0;
    static int cached_pb = 0;
    static int cached_lpl = 0;

    bool need_rebuild = (n_pages != cached_n_pages || pb != cached_pb);
    bool need_lpl_update = (last_page_len != cached_lpl);

    if (need_rebuild) {
        std::vector<int32_t> h_indptr(pb + 1);
        std::vector<int32_t> h_indices(pb * n_pages);
        for (int b = 0; b <= pb; b++) h_indptr[b] = b * n_pages;
        for (int b = 0; b < pb; b++)
            for (int p = 0; p < n_pages; p++)
                h_indices[b * n_pages + p] = p;
        cudaMemcpy(g_meta.d_indptr, h_indptr.data(), (pb + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(g_meta.d_indices, h_indices.data(), pb * n_pages * sizeof(int32_t), cudaMemcpyHostToDevice);
        cached_n_pages = n_pages;
        cached_pb = pb;
        need_lpl_update = true;  // rebuild includes lpl
    }

    if (need_lpl_update) {
        std::vector<int32_t> h_lpl(pb, last_page_len);
        cudaMemcpy(g_meta.d_last_page_len, h_lpl.data(), pb * sizeof(int32_t), cudaMemcpyHostToDevice);
        cached_lpl = last_page_len;
    }

    // paged_kv_t strides for NHD layout with PAGE_SIZE=16
    int64_t kv_strides[3] = {
        (int64_t)FI_PAGE_SIZE * n_head_kv * HEAD_DIM,  // stride_page
        (int64_t)n_head_kv * HEAD_DIM,                  // stride_n
        (int64_t)HEAD_DIM                                // stride_h
    };

    paged_kv_t<half, int32_t> paged_kv(
        n_head_kv, FI_PAGE_SIZE, HEAD_DIM, n_tokens,
        QKVLayout::kNHD,
        k, v, kv_strides,
        g_meta.d_indices, g_meta.d_indptr, g_meta.d_last_page_len,
        nullptr);

    BatchDecodeParams<half, half, half, int32_t> params(
        const_cast<half *>(q),
        nullptr,  // q_rope_offset
        paged_kv,
        output,
        nullptr, nullptr,
        n_head_q,
        n_head_q * HEAD_DIM,
        HEAD_DIM,
        -1, 0.0f, scale, 1.0f, 10000.0f);

    params.padded_batch_size = pb;
    params.request_indices = g_meta.d_request_indices;
    params.kv_tile_indices = g_meta.d_kv_tile_indices;
    params.o_indptr = g_meta.d_o_indptr;
    params.kv_chunk_size_ptr = g_meta.d_kv_chunk_size;
    params.block_valid_mask = nullptr;
    params.partition_kv = false;

    return BatchDecodeWithPagedKVCacheDispatched<
        HEAD_DIM,
        PosEncodingMode::kNone,
        StdAttention,
        BatchDecodeParams<half, half, half, int32_t>
    >(params, nullptr, nullptr, false, stream);
}

extern "C" cudaError_t launch_contiguous_decode_attention(
    half * output, const half * q, const half * k, const half * v,
    int n_tokens, int kv_len, int n_head_q, int n_head_kv, int head_dim,
    float scale, cudaStream_t stream)
{
    switch (head_dim) {
        case 64:
            return batch_decode_paged_impl<64>(output, q, (half*)k, (half*)v,
                n_tokens, kv_len, n_head_q, n_head_kv, scale, stream);
        case 128:
            return batch_decode_paged_impl<128>(output, q, (half*)k, (half*)v,
                n_tokens, kv_len, n_head_q, n_head_kv, scale, stream);
        default:
            fprintf(stderr, "[fi-attn] unsupported head_dim=%d\n", head_dim);
            return cudaErrorInvalidValue;
    }
}

extern "C" void launch_f32_to_f16(half *, const float *, int64_t, cudaStream_t);
extern "C" void launch_f16_to_f32(float *, const half *, int, cudaStream_t);
extern "C" void launch_transpose_thd_to_dth_f16(half *, const half *, int, int, int, cudaStream_t);
extern "C" void launch_transpose_thd_to_dth_f16_to_f32(float *, const half *, int, int, int, cudaStream_t);

// Temp buffer for FlashInfer output (before transpose)
static half * d_fi_tmp = nullptr;
static int d_fi_tmp_size = 0;

static void ensure_fi_tmp(int n) {
    if (d_fi_tmp_size >= n) return;
    if (d_fi_tmp) cudaFree(d_fi_tmp);
    d_fi_tmp_size = std::max(n, 16 * 128 * 128);
    cudaMalloc(&d_fi_tmp, d_fi_tmp_size * sizeof(half));
}

extern "C" int flashinfer_attn_kernel_wrapper(
    void * output, const void * q, const void * k, const void * v,
    int n_tokens, int kv_len, int n_head_q, int n_head_kv, int head_dim,
    float scale, int q_type, void * stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    int total = n_tokens * n_head_q * head_dim;

    if (q_type == 1) {
        // FP16 model: Q is FP16, output is FP16
        if (n_tokens == 1) {
            // Decode: no transpose needed (T=1 -> layouts identical)
            return (int)launch_contiguous_decode_attention(
                (half *)output, (const half *)q, (const half *)k, (const half *)v,
                n_tokens, kv_len, n_head_q, n_head_kv, head_dim, scale, s);
        }
        // Prefill: FlashInfer writes [T,H,D], ggml expects [D,T,H] -> transpose
        ensure_fi_tmp(total);
        cudaError_t err = launch_contiguous_decode_attention(
            d_fi_tmp, (const half *)q, (const half *)k, (const half *)v,
            n_tokens, kv_len, n_head_q, n_head_kv, head_dim, scale, s);
        if (err != cudaSuccess) return (int)err;
        launch_transpose_thd_to_dth_f16((half *)output, d_fi_tmp,
                                         n_tokens, n_head_q, head_dim, s);
        return 0;
    }

    // Quantized model: Q is F32, output is F32
    // Convert Q F32->FP16, run FlashInfer, transpose+convert FP16->F32
    static half * d_q_fp16 = nullptr;
    static int q_buf_size = 0;
    if (total > q_buf_size) {
        if (d_q_fp16) cudaFree(d_q_fp16);
        q_buf_size = std::max(total, 16 * 128 * 128);
        cudaMalloc(&d_q_fp16, q_buf_size * sizeof(half));
    }
    ensure_fi_tmp(total);

    launch_f32_to_f16(d_q_fp16, (const float *)q, total, s);

    cudaError_t err = launch_contiguous_decode_attention(
        d_fi_tmp, d_q_fp16, (const half *)k, (const half *)v,
        n_tokens, kv_len, n_head_q, n_head_kv, head_dim, scale, s);
    if (err != cudaSuccess) return (int)err;

    if (n_tokens == 1) {
        // Decode: no transpose, just convert FP16->F32
        launch_f16_to_f32((float *)output, d_fi_tmp, total, s);
    } else {
        // Prefill: fused transpose [T,H,D]->[D,T,H] + FP16->F32
        launch_transpose_thd_to_dth_f16_to_f32((float *)output, d_fi_tmp,
                                                 n_tokens, n_head_q, head_dim, s);
    }
    return 0;
}

// Matmul Hook: Dequant Q4_K -> FP16 temp + cuBLAS Tensor Core
// For batch>1, dequant weight to temp FP16 buffer and use cuBLAS.
// cuBLAS Tensor Core matmul scales better than ggml's dp4a MMVQ at M>1.

#include <cublas_v2.h>

extern "C" void launch_dequant_q4k(half *, const void *, int, cudaStream_t);
extern "C" void launch_dequant_q8_0(half *, const void *, int, cudaStream_t);
extern "C" void launch_dequant_q5_0(half *, const void *, int, cudaStream_t);
extern "C" void launch_dequant_q6k(half *, const void *, int, cudaStream_t);

static cublasHandle_t g_cublas = nullptr;
static half * g_weight_buf = nullptr;
static size_t g_weight_buf_size = 0;
static half * g_input_buf = nullptr;   // F32->FP16 converted input
static half * g_output_buf = nullptr;  // FP16 output (before F16->F32)
static size_t g_io_buf_size = 0;

static void ensure_cublas() {
    if (!g_cublas) {
        cublasCreate(&g_cublas);
        cublasSetMathMode(g_cublas, CUBLAS_TENSOR_OP_MATH);
    }
}

static void ensure_weight_buf(size_t n_elements) {
    if (g_weight_buf_size >= n_elements) return;
    // Sync before free to prevent in-flight kernel reads on freed memory
    cudaDeviceSynchronize();
    if (g_weight_buf) cudaFree(g_weight_buf);
    g_weight_buf_size = std::max(n_elements, (size_t)(8192 * 8192));  // pre-alloc large
    cudaMalloc(&g_weight_buf, g_weight_buf_size * sizeof(half));
}

static void ensure_io_buf(size_t n_elements) {
    if (g_io_buf_size >= n_elements) return;
    cudaDeviceSynchronize();
    if (g_input_buf) cudaFree(g_input_buf);
    if (g_output_buf) cudaFree(g_output_buf);
    g_io_buf_size = std::max(n_elements, (size_t)(16 * 8192));  // pre-alloc large
    cudaMalloc(&g_input_buf, g_io_buf_size * sizeof(half));
    cudaMalloc(&g_output_buf, g_io_buf_size * sizeof(half));
}

extern "C" int flashinfer_matmul_hook(
    void * dst_ptr, const void * src0_ptr, const void * src1_ptr,
    int M, int N, int K, int src0_type, void * stream_ptr)
{
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Disabled (returns -1 to fall back to ggml for quantized matmul): cuBLAS on FP16-dequantized
    // weights rounds differently from ggml's INT8 dp4a MMVQ, and the difference compounds across
    // layers into visible output degradation.
    return -1;

    ensure_cublas();
    cublasSetStream(g_cublas, stream);

    size_t n_elements = (size_t)N * K;
    ensure_weight_buf(n_elements);

    // Dequant weight to FP16 temp buffer
    switch (src0_type) {
        case 12: launch_dequant_q4k(g_weight_buf, src0_ptr, (int)n_elements, stream); break;
        case  8: launch_dequant_q8_0(g_weight_buf, src0_ptr, (int)n_elements, stream); break;
        case  6: launch_dequant_q5_0(g_weight_buf, src0_ptr, (int)n_elements, stream); break;
        case 14: launch_dequant_q6k(g_weight_buf, src0_ptr, (int)n_elements, stream); break;
        default: return -1;  // unsupported type, fall through to ggml
    }

    // cuBLAS matmul: dst[M,N] = src1[M,K] x weight[N,K]^T
    // ggml passes FP32 activations; convert the input F32 -> FP16
    size_t in_size = (size_t)M * K;
    size_t out_size = (size_t)M * N;
    ensure_io_buf(std::max(in_size, out_size));
    launch_f32_to_f16(g_input_buf, (const float *)src1_ptr, (int)in_size, stream);

    // cuBLAS GemmEx: FP16 x FP16 -> F32 (Tensor Core with FP32 accumulation)
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(g_cublas,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 g_weight_buf, CUDA_R_16F, K,
                 g_input_buf, CUDA_R_16F, K,
                 &beta,
                 dst_ptr, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS) {
        static bool err_logged = false;
        if (!err_logged) {
            fprintf(stderr, "[cublas-matmul] GemmEx FAILED status=%d M=%d N=%d K=%d\n",
                    (int)status, M, N, K);
            err_logged = true;
        }
        return -1;
    }

    static int call_count = 0;
    if (call_count < 10) {
        cudaStreamSynchronize(stream);
        float out[2];
        cudaMemcpy(out, dst_ptr, sizeof(out), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[cublas-matmul] #%d M=%d N=%d K=%d t=%d out=%.4f,%.4f\n",
                call_count, M, N, K, src0_type, out[0], out[1]);
    }
    call_count++;

    return 0;
}

#endif // ALPHA_HAS_FLASHINFER
