#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_set_flashinfer_hook(void * fn);
void ggml_cuda_set_matmul_hook(void * fn);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);
