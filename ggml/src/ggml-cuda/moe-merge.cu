// ALPHA_MOE_MERGE: fused MoE output merge. See moe-merge.cuh.
// Compiled with -fmad=false (CMakeLists.txt): the op chain it replaces multiplies in one kernel and adds in
// another, so a contracted fma(e, w, s) here would not be bitwise equal to it.
#include "moe-merge.cuh"

#define MOE_MERGE_MAX_ADD 4
#define MOE_MERGE_BLOCK_SIZE 256

static __global__ void k_moe_merge(
        const float * __restrict__ e, const float * __restrict__ w, float * __restrict__ dst,
        const float * a0, const float * a1, const float * a2, const float * a3,
        const int n_add, const int k, const int64_t n_embd, const int64_t n_elem,
        const int64_t se0, const int64_t se1, const int64_t se2,
        const int64_t sw1, const int64_t sw2,
        const int64_t sa0_0, const int64_t sa1_0, const int64_t sa0_1, const int64_t sa1_1,
        const int64_t sa0_2, const int64_t sa1_2, const int64_t sa0_3, const int64_t sa1_3) {
    const int64_t idx = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_elem) {
        return;
    }
    const int64_t t = idx / n_embd;
    const int64_t d = idx - t * n_embd;

    const int64_t eb = d * se0 + t * se2;
    const int64_t wb = t * sw2;

    float s = e[eb] * w[wb];
    for (int j = 1; j < k; ++j) {
        const float p = e[eb + j * se1] * w[wb + j * sw1];
        s = s + p;
    }
    if (n_add > 0) { s = s + a0[d * sa0_0 + t * sa1_0]; }
    if (n_add > 1) { s = s + a1[d * sa0_1 + t * sa1_1]; }
    if (n_add > 2) { s = s + a2[d * sa0_2 + t * sa1_2]; }
    if (n_add > 3) { s = s + a3[d * sa0_3 + t * sa1_3]; }
    dst[idx] = s;
}

void ggml_cuda_op_moe_merge(ggml_backend_cuda_context & ctx,
                            const ggml_tensor * experts, const ggml_tensor * weights,
                            const ggml_tensor * const * addends, int n_add,
                            ggml_tensor * dst) {
    GGML_ASSERT(n_add >= 0 && n_add <= MOE_MERGE_MAX_ADD);
    GGML_ASSERT(experts->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd = experts->ne[0];
    const int     k      = (int) experts->ne[1];
    const int64_t T      = experts->ne[2];
    GGML_ASSERT(dst->ne[0] == n_embd && dst->ne[1] == T);

    const int64_t fs = (int64_t) sizeof(float);
    const float * ap[MOE_MERGE_MAX_ADD] = { nullptr, nullptr, nullptr, nullptr };
    int64_t s0[MOE_MERGE_MAX_ADD] = { 0, 0, 0, 0 };
    int64_t s1[MOE_MERGE_MAX_ADD] = { 0, 0, 0, 0 };
    for (int i = 0; i < n_add; ++i) {
        GGML_ASSERT(addends[i]->type == GGML_TYPE_F32);
        ap[i] = (const float *) addends[i]->data;
        s0[i] = (int64_t) addends[i]->nb[0] / fs;
        s1[i] = (int64_t) addends[i]->nb[1] / fs;
    }

    const int64_t n_elem = n_embd * T;
    const int     bs     = MOE_MERGE_BLOCK_SIZE;
    const int64_t nb     = (n_elem + bs - 1) / bs;
    const dim3    grid((unsigned) nb, 1, 1);
    const dim3    blk((unsigned) bs, 1, 1);

    k_moe_merge<<<grid, blk, 0, ctx.stream()>>>(
        (const float *) experts->data, (const float *) weights->data, (float *) dst->data,
        ap[0], ap[1], ap[2], ap[3],
        n_add, k, n_embd, n_elem,
        (int64_t) experts->nb[0] / fs, (int64_t) experts->nb[1] / fs, (int64_t) experts->nb[2] / fs,
        (int64_t) weights->nb[1] / fs, (int64_t) weights->nb[2] / fs,
        s0[0], s1[0], s0[1], s1[1], s0[2], s1[2], s0[3], s1[3]);
    CUDA_CHECK(cudaGetLastError());
}
