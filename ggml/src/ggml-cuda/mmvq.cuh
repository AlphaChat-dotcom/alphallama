#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// Per-graph cache of the q8_1-quantized activation (ALPHA_Q8_CACHE). Every MMVQ GEMV first quantizes its
// src1; at decode the router, shared-expert and gate GEMVs of a layer take the same normed hidden state, so
// the copy is shared until something writes over that region. begin_graph() is called at the start of each
// backend graph compute, note_write() for every evaluated node (a dst overlapping the cached src1 invalidates it).
void ggml_cuda_mmvq_q8_cache_begin_graph();
void ggml_cuda_mmvq_q8_cache_note_write(const void * data, size_t nbytes);

// mmid_tokens_as_samples (ALPHA_MMID_BI): internal, set by the multi-token MUL_MAT_ID path when it maps the
// tokens onto the kernel's sample axis so the 1-token column geometry (and its arithmetic) is kept.
void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr, bool mmid_tokens_as_samples = false);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
