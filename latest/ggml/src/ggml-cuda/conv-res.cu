#include "conv-res.cuh"

// one thread per channel and sequence, the tokens in order: the sum as ssm_conv_f32 forms it, then the residual add
// and the scale, each rounded as its own op
template <int d_conv>
static __global__ void conv_res_f32(const float * __restrict__ x, const float * __restrict__ w, const float * cur, float * dst,
        const int64_t x_s1, const int64_t x_s2, const int64_t w_s1, const int64_t cur_s1, const int64_t dst_s1,
        const int nr, const int n_t, const float b, const float scale, const float bias) {
    const int ch = blockIdx.y*blockDim.x + threadIdx.x;
    const int s  = blockIdx.x;

    ggml_cuda_pdl_sync();
    if (ch >= nr) {
        return;
    }

    float wv[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wv[j] = w[ch*w_s1 + j];
    }

    const float * xb = x + s*x_s2 + ch*x_s1;
    for (int i = 0; i < n_t; ++i) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sumf += xb[i + j]*wv[j];
        }
        sumf += b;

        const int64_t tok = int64_t(s)*n_t + i;
        const float   v   = __fadd_rn(cur[tok*cur_s1 + ch], sumf);
        dst[tok*dst_s1 + ch] = scale*v + bias;
    }
    ggml_cuda_pdl_lc();
}

void ggml_cuda_op_conv_res_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * conv, const ggml_tensor * cur,
        float scale, float bias, ggml_tensor * dst, bool copy_x) {
    const ggml_tensor * x = conv->src[0];
    const ggml_tensor * w = conv->src[1];

    // the output may take the memory of x, which the threads read across tokens: read a copy then
    ggml_cuda_pool_alloc<char> x_copy(ctx.pool());
    const float * xd = (const float *) x->data;
    if (copy_x) {
        x_copy.alloc(ggml_nbytes(x));
        CUDA_CHECK(cudaMemcpyAsync(x_copy.get(), x->data, ggml_nbytes(x), cudaMemcpyDeviceToDevice, ctx.stream()));
        xd = (const float *) x_copy.get();
    }

    const int64_t d_conv = w->ne[0];
    const int64_t nr     = conv->ne[0];
    const int64_t n_t    = conv->ne[1];
    const int64_t n_s    = conv->ne[2];

    const int threads = 128;
    const ggml_cuda_kernel_launch_params params = ggml_cuda_kernel_launch_params(
            dim3(n_s, (nr + threads - 1)/threads, 1), dim3(threads, 1, 1), 0, ctx.stream());

    const int64_t x_s1   = x->nb[1]/sizeof(float);
    const int64_t x_s2   = x->nb[2]/sizeof(float);
    const int64_t w_s1   = w->nb[1]/sizeof(float);
    const int64_t cur_s1 = cur->nb[1]/sizeof(float);
    const int64_t dst_s1 = dst->nb[1]/sizeof(float);

    const float * wd = (const float *) w->data;
    const float * cd = (const float *) cur->data;
    float       * dd = (float *) dst->data;

    switch (d_conv) {
        case 4: ggml_cuda_kernel_launch(conv_res_f32<4>, params, xd, wd, cd, dd, x_s1, x_s2, w_s1, cur_s1, dst_s1, (int) nr, (int) n_t, 0.0f, scale, bias); break;
        case 3: ggml_cuda_kernel_launch(conv_res_f32<3>, params, xd, wd, cd, dd, x_s1, x_s2, w_s1, cur_s1, dst_s1, (int) nr, (int) n_t, 0.0f, scale, bias); break;
        default: GGML_ABORT("fatal error");
    }
}
