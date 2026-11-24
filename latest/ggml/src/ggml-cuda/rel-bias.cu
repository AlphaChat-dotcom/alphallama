#include "rel-bias.cuh"
#include "convert.cuh"

#include "ggml-rel-bias.h"

// one block per (head, mask row, key chunk): R[:, h, row] and the sink in registers, each key's bias computed from its bucket
// column of W (bucket n_bucket, one past the table, has a zero bias), minus the sink, plus the mask row, rounded to F16 as the
// cast after the add
#define REL_BIAS_MAX_REL 32
template <typename mask_t, int n_rel>
static __global__ void rel_bias_mask_f16(
        const float * __restrict__ W, const float * __restrict__ R, const float * __restrict__ sinks,
        const int32_t * __restrict__ rel_pos, const mask_t * __restrict__ mask, half * __restrict__ dst,
        const int n_bucket, const int n_kv, const int nr, const int k_chunk,
        const int64_t r_s1, const int64_t r_s2, const int64_t s_s1, const int64_t p_s1,
        const int64_t m_s1, const int64_t m_s3, const int64_t d_s1, const int64_t d_s2, const int64_t d_s3) {
    const int h   = blockIdx.x;
    const int row = blockIdx.y;           // t + s*nr
    const int t   = row % nr;
    const int s   = row / nr;
    const int k0  = blockIdx.z*k_chunk;
    const int k1  = min(n_kv, k0 + k_chunk);

    ggml_cuda_pdl_sync();

    const float   sink = sinks[row*s_s1 + h];
    const float * rp   = R + row*r_s2 + h*r_s1;
    float r[n_rel];
#pragma unroll
    for (int i = 0; i < n_rel; ++i) {
        r[i] = rp[i];
    }

    const int32_t * pos = rel_pos + row*p_s1;
    const mask_t  * m   = mask + s*m_s3 + t*m_s1;
    half          * d   = dst + s*d_s3 + h*d_s2 + t*d_s1;
    for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
        const int b = min(max(pos[k], 0), n_bucket);
        float v = 0.0f;
        if (b < n_bucket) {
            const float * w = W + (int64_t) b*n_rel;
#pragma unroll
            for (int i = 0; i < n_rel; ++i) {
                v += w[i]*r[i];
            }
        }
        d[k] = __float2half((v - sink) + ggml_cuda_cast<float>(m[k]));
    }
    ggml_cuda_pdl_lc();
}

bool ggml_cuda_rel_bias_mask_supported(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) {
        return false;
    }
    ggml_custom_op_params p;
    memcpy(&p, op->op_params, sizeof(p));
    if (p.fun != ggml_rel_bias_mask_cpu) {
        return false;
    }
    const ggml_tensor * W    = op->src[0];
    const ggml_tensor * mask = op->src[4];
    // the bias column in shared memory
    return (W->ne[0] == 8 || W->ne[0] == 16 || W->ne[0] == 32) && (mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
}

void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * W       = dst->src[0];
    const ggml_tensor * R       = dst->src[1];
    const ggml_tensor * sinks   = dst->src[2];
    const ggml_tensor * rel_pos = dst->src[3];
    const ggml_tensor * mask    = dst->src[4];

    const int n_rel    = (int) W->ne[0];
    const int n_bucket = (int) W->ne[1];
    const int n_kv     = (int) dst->ne[0];
    const int nr       = (int) dst->ne[1];
    const int n_head   = (int) dst->ne[2];
    const int n_stream = (int) dst->ne[3];

    const int k_chunk  = 1024;
    const int threads  = 128;
    const dim3 grid(n_head, nr*n_stream, (n_kv + k_chunk - 1)/k_chunk);

    const int64_t r_s1 = R->nb[1]/sizeof(float),       r_s2 = R->nb[2]/sizeof(float);
    const int64_t s_s1 = sinks->nb[1]/sizeof(float);
    const int64_t p_s1 = rel_pos->nb[1]/sizeof(int32_t);
    const int64_t d_s1 = dst->nb[1]/sizeof(half),      d_s2 = dst->nb[2]/sizeof(half), d_s3 = dst->nb[3]/sizeof(half);

    const ggml_cuda_kernel_launch_params params = ggml_cuda_kernel_launch_params(grid, dim3(threads, 1, 1), 0, ctx.stream());

    const float   * Wd = (const float *) W->data;
    const float   * Rd = (const float *) R->data;
    const float   * Sd = (const float *) sinks->data;
    const int32_t * Pd = (const int32_t *) rel_pos->data;
    half          * Dd = (half *) dst->data;

    const int64_t m_es = ggml_element_size(mask);
    const int64_t m_s1 = mask->nb[1]/m_es, m_s3 = mask->nb[3]/m_es;
#define REL_BIAS_LAUNCH(MT, NR) ggml_cuda_kernel_launch(rel_bias_mask_f16<MT, NR>, params, Wd, Rd, Sd, Pd, (const MT *) mask->data, Dd,             n_bucket, n_kv, nr, k_chunk, r_s1, r_s2, s_s1, p_s1, m_s1, m_s3, d_s1, d_s2, d_s3)
    if (mask->type == GGML_TYPE_F16) {
        switch (n_rel) {
            case 8:  REL_BIAS_LAUNCH(half, 8);  break;
            case 16: REL_BIAS_LAUNCH(half, 16); break;
            default: REL_BIAS_LAUNCH(half, 32); break;
        }
    } else {
        switch (n_rel) {
            case 8:  REL_BIAS_LAUNCH(float, 8);  break;
            case 16: REL_BIAS_LAUNCH(float, 16); break;
            default: REL_BIAS_LAUNCH(float, 32); break;
        }
    }
#undef REL_BIAS_LAUNCH
}
