#include "common.cuh"
#include "ggml.h"

// DeepSeek V4 CUDA kernels - stride-aware memory access
// All reads use byte strides (nb) from tensor metadata.

#define DSV4_F32(ptr, byte_off) (*(const float *)((const char *)(ptr) + (byte_off)))
#define DSV4_F32W(ptr, byte_off) (*(float *)((char *)(ptr) + (byte_off)))

// HC weighted sum: x[n_embd, n_hc, n_tokens] * w[n_hc, n_tokens] -> [n_embd, n_tokens]
static __global__ void dsv4_hc_weighted_sum_k(
        const void * x, const void * w, void * dst,
        int64_t n_embd, int64_t n_hc, int64_t n_tokens,
        size_t x_nb0, size_t x_nb1, size_t x_nb2,
        size_t w_nb0, size_t w_nb1,
        size_t d_nb0, size_t d_nb1) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd * n_tokens) return;
    const int64_t d = i % n_embd;
    const int64_t t = i / n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < n_hc; ++h) {
        acc += DSV4_F32(x, d*x_nb0 + h*x_nb1 + t*x_nb2)
             * DSV4_F32(w, h*w_nb0 + t*w_nb1);
    }
    DSV4_F32W(dst, d*d_nb0 + t*d_nb1) = acc;
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * w = dst->src[1];
    const int64_t n_embd = dst->ne[0];
    const int64_t n_hc = x->ne[1];
    const int64_t n_tokens = dst->ne[1];
    const int bs = 256;
    dsv4_hc_weighted_sum_k<<<(n_embd*n_tokens+bs-1)/bs, bs, 0, ctx.stream()>>>(
        x->data, w->data, dst->data, n_embd, n_hc, n_tokens,
        x->nb[0], x->nb[1], x->nb[2], w->nb[0], w->nb[1], dst->nb[0], dst->nb[1]);
}

// HC expand: block_out[n_embd, n_tokens] + post*block + comb*residual -> [n_embd, n_hc, n_tokens]
static __global__ void dsv4_hc_expand_k(
        const void * bo, const void * res, const void * post, const void * comb, void * dst,
        int64_t n_embd, int64_t n_hc, int64_t n_tokens,
        size_t bo_nb0, size_t bo_nb1,
        size_t r_nb0, size_t r_nb1, size_t r_nb2,
        size_t p_nb0, size_t p_nb1,
        size_t c_nb0, size_t c_nb1, size_t c_nb2,
        size_t d_nb0, size_t d_nb1, size_t d_nb2) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd * n_hc * n_tokens) return;
    const int64_t d = i % n_embd;
    const int64_t dh = (i / n_embd) % n_hc;
    const int64_t t = i / (n_embd * n_hc);

    float acc = DSV4_F32(bo, d*bo_nb0 + t*bo_nb1) * DSV4_F32(post, dh*p_nb0 + t*p_nb1);
    for (int64_t sh = 0; sh < n_hc; ++sh) {
        acc += DSV4_F32(comb, dh*c_nb0 + sh*c_nb1 + t*c_nb2)
             * DSV4_F32(res, d*r_nb0 + sh*r_nb1 + t*r_nb2);
    }
    DSV4_F32W(dst, d*d_nb0 + dh*d_nb1 + t*d_nb2) = acc;
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * bo = dst->src[0];
    const ggml_tensor * res = dst->src[1];
    const ggml_tensor * post = dst->src[2];
    const ggml_tensor * comb = dst->src[3];
    const int64_t ne = dst->ne[0] * dst->ne[1] * dst->ne[2];
    const int bs = 256;
    dsv4_hc_expand_k<<<(ne+bs-1)/bs, bs, 0, ctx.stream()>>>(
        bo->data, res->data, post->data, comb->data, dst->data,
        dst->ne[0], dst->ne[1], dst->ne[2],
        bo->nb[0], bo->nb[1],
        res->nb[0], res->nb[1], res->nb[2],
        post->nb[0], post->nb[1],
        comb->nb[0], comb->nb[1], comb->nb[2],
        dst->nb[0], dst->nb[1], dst->nb[2]);
}

// HC split sinkhorn: per-token sigmoid+sinkhorn on mix logits
static __global__ void dsv4_hc_split_sinkhorn_k(
        const void * mixes, const void * scale, const void * base, void * dst,
        int64_t n_mix, int64_t n_tokens, int n_hc, int sinkhorn_iters, float eps,
        size_t m_nb0, size_t m_nb1, size_t d_nb0, size_t d_nb1) {
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens) return;

    const float ps = DSV4_F32(scale, 0);
    const float pos = DSV4_F32(scale, 4);
    const float cs = DSV4_F32(scale, 8);
    const float * b = (const float *)base;

    float m[256];
    for (int i = 0; i < (int)n_mix; ++i)
        m[i] = DSV4_F32(mixes, i*m_nb0 + t*m_nb1);

    // Pre: sigmoid
    for (int i = 0; i < n_hc; ++i) {
        float z = m[i] * ps + b[i];
        DSV4_F32W(dst, i*d_nb0 + t*d_nb1) = 1.0f/(1.0f + expf(-z)) + eps;
    }
    // Post: 2*sigmoid
    for (int i = 0; i < n_hc; ++i) {
        int off = n_hc + i;
        float z = m[off] * pos + b[off];
        DSV4_F32W(dst, off*d_nb0 + t*d_nb1) = 2.0f/(1.0f + expf(-z));
    }
    // Comb: softmax + sinkhorn
    float c[16*16];
    for (int dh = 0; dh < n_hc; ++dh) {
        float mx = -INFINITY;
        for (int sh = 0; sh < n_hc; ++sh) {
            int idx = sh + dh*n_hc, off = 2*n_hc + idx;
            c[idx] = m[off]*cs + b[off];
            mx = fmaxf(mx, c[idx]);
        }
        float s = 0;
        for (int sh = 0; sh < n_hc; ++sh) { int idx=sh+dh*n_hc; c[idx]=expf(c[idx]-mx); s+=c[idx]; }
        for (int sh = 0; sh < n_hc; ++sh) { int idx=sh+dh*n_hc; c[idx]=c[idx]/s+eps; }
    }
    // Sinkhorn must match reference kernel.py hc_split_sinkhorn exactly:
    //   after softmax(+eps): one column-normalize, then (iters-1) x (row-norm, col-norm),
    //   every division uses (sum + eps), and the sequence ends on a column-normalize.
    // comb[i,j] mixes residual stream i into output stream j (sum_i comb[i,j]*res[i]), so the
    // final matrix must be column-normalized; ending row-normalized gives wrong HC residual mixing.
    // Column-normalize (sum over dst-stream dh for each src-stream sh):
    for (int sh = 0; sh < n_hc; ++sh) {
        float s=0; for (int dh=0;dh<n_hc;++dh) s+=c[sh+dh*n_hc];
        float inv=1.0f/(s+eps); for (int dh=0;dh<n_hc;++dh) c[sh+dh*n_hc]*=inv;
    }
    for (int it = 0; it < sinkhorn_iters - 1; ++it) {
        // Row-normalize (sum over sh for each dh):
        for (int dh = 0; dh < n_hc; ++dh) {
            float s=0; for (int sh=0;sh<n_hc;++sh) s+=c[sh+dh*n_hc];
            float inv=1.0f/(s+eps); for (int sh=0;sh<n_hc;++sh) c[sh+dh*n_hc]*=inv;
        }
        // Column-normalize (sum over dh for each sh):
        for (int sh = 0; sh < n_hc; ++sh) {
            float s=0; for (int dh=0;dh<n_hc;++dh) s+=c[sh+dh*n_hc];
            float inv=1.0f/(s+eps); for (int dh=0;dh<n_hc;++dh) c[sh+dh*n_hc]*=inv;
        }
    }
    for (int i = 0; i < n_hc*n_hc; ++i)
        DSV4_F32W(dst, (2*n_hc+i)*d_nb0 + t*d_nb1) = c[i];
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base = dst->src[2];
    const int n_hc = ((int32_t *)dst->op_params)[0];
    const int si = ((int32_t *)dst->op_params)[1];
    float eps; memcpy(&eps, (int32_t *)dst->op_params + 2, sizeof(float));
    const int64_t n_mix = mixes->ne[0], n_tokens = mixes->ne[1];
    const int bs = 256;
    dsv4_hc_split_sinkhorn_k<<<(n_tokens+bs-1)/bs, bs, 0, ctx.stream()>>>(
        mixes->data, scale->data, base->data, dst->data,
        n_mix, n_tokens, n_hc, si, eps,
        mixes->nb[0], mixes->nb[1], dst->nb[0], dst->nb[1]);
}
