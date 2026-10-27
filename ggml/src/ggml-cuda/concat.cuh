#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Fused CONCAT(dim 0) + CPY of the concat result's last k columns into a contiguous state buffer
// (the recurrent conv-state roll of DeltaNet / Mamba decode). `concat` is the CONCAT node, `cpy` the CPY node.
void ggml_cuda_op_concat_shift(ggml_backend_cuda_context & ctx, ggml_tensor * concat, ggml_tensor * cpy);
