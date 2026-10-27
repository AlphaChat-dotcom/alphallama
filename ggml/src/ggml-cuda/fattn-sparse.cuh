#pragma once

#include "common.cuh"

// ALPHA_QSA_FA_SPARSE: decode flash attention over a listed subset of KV cells (dst->src[5], I32 [n_sel, n_q]),
// reading K/V in place from the cache. See fattn-sparse.cu.
bool ggml_cuda_flash_attn_sparse_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
