#include "concat-store.cuh"

// one thread per element of x = concat(a, b) along dim 0: x is written contiguously, and an element in columns
// [off, off + n_cols) is also written to the store, at its index in the copied view flattened (as ggml_cpy maps elements)
static __global__ void concat_store_f32(
        const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ x, float * __restrict__ st,
        const int ne0a, const int ne0, const int ne1, const int64_t n,
        const int64_t a_s0, const int64_t a_s1, const int64_t a_s2,
        const int64_t b_s0, const int64_t b_s1, const int64_t b_s2,
        const int off, const int n_cols, const int st_ne0, const int st_ne1,
        const int64_t st_s0, const int64_t st_s1, const int64_t st_s2) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    ggml_cuda_pdl_sync();
    if (idx >= n) {
        return;
    }

    const int     i0 = idx % ne0;
    const int64_t r  = idx / ne0;
    const int     i1 = r % ne1;
    const int     i2 = r / ne1;

    const float v = i0 < ne0a ? a[i0*a_s0 + i1*a_s1 + i2*a_s2] : b[(i0 - ne0a)*b_s0 + i1*b_s1 + i2*b_s2];
    x[idx] = v;

    if (i0 >= off && i0 < off + n_cols) {
        int64_t f = (i0 - off) + (int64_t) n_cols*(i1 + (int64_t) ne1*i2);
        const int j0 = f % st_ne0;
        f /= st_ne0;
        const int j1 = f % st_ne1;
        const int j2 = f / st_ne1;
        st[j0*st_s0 + j1*st_s1 + j2*st_s2] = v;
    }
    ggml_cuda_pdl_lc();
}

void ggml_cuda_op_concat_store_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * x, int64_t off, int64_t n_cols, ggml_tensor * st) {
    const ggml_tensor * a = x->src[0];
    const ggml_tensor * b = x->src[1];

    const int64_t n = ggml_nelements(x);
    const int threads = 256;
    const ggml_cuda_kernel_launch_params params = ggml_cuda_kernel_launch_params(
            dim3((n + threads - 1)/threads, 1, 1), dim3(threads, 1, 1), 0, ctx.stream());

    const int64_t f = sizeof(float);
    ggml_cuda_kernel_launch(concat_store_f32, params,
            (const float *) a->data, (const float *) b->data, (float *) x->data, (float *) st->data,
            (int) a->ne[0], (int) x->ne[0], (int) x->ne[1], n,
            (int64_t) (a->nb[0]/f), (int64_t) (a->nb[1]/f), (int64_t) (a->nb[2]/f),
            (int64_t) (b->nb[0]/f), (int64_t) (b->nb[1]/f), (int64_t) (b->nb[2]/f),
            (int) off, (int) n_cols, (int) st->ne[0], (int) st->ne[1],
            (int64_t) (st->nb[0]/f), (int64_t) (st->nb[1]/f), (int64_t) (st->nb[2]/f));
}
