#include "depth-mix.cuh"

#define DEPTH_MIX_MAX_ROWS  (GGML_MAX_SRC - 1)
#define DEPTH_MIX_MAX_QUERY 4
#define DEPTH_MIX_BLOCK     512

struct depth_mix_rows {
    const float * x[DEPTH_MIX_MAX_ROWS]; // row r of token t at x[r] + t*s[r]: the newest block first, cur last
    int64_t       s[DEPTH_MIX_MAX_ROWS];
};

template <int width> struct depth_mix_vec;
template <> struct depth_mix_vec<1> { typedef float  type; };
template <> struct depth_mix_vec<4> { typedef float4 type; };

template <int width>
static __device__ __forceinline__ float depth_mix_get(const typename depth_mix_vec<width>::type & v, int c) {
    if constexpr (width == 1) {
        return v;
    } else {
        return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
    }
}

// one block per token: the RMS norm of every row and its scores against every query in one pass, the softmax over
// the rows, then the weighted sum of the normalized rows; width 4 reads float4 (n and the strides multiples of 4)
template <int n_query, int width>
static __global__ void __launch_bounds__(DEPTH_MIX_BLOCK) depth_mix_f32(const depth_mix_rows rows, const int n_rows,
        const float * __restrict__ query, const int64_t q_s1, const int64_t q_s2, float * __restrict__ dst, const int n,
        const float eps, const float scale) {
    typedef typename depth_mix_vec<width>::type vec;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_acc     = DEPTH_MIX_MAX_ROWS*(n_query + 1);

    const int64_t t   = blockIdx.x;
    const int     tid = threadIdx.x;
    const int     nv  = n/width;

    ggml_cuda_pdl_sync();

    const float * q = query + t*q_s2;

    // per row: the sum of squares, then the dot product with each query
    float acc[n_acc] = {0.0f};
    for (int iv = tid; iv < nv; iv += DEPTH_MIX_BLOCK) {
        vec qv[n_query];
#pragma unroll
        for (int k = 0; k < n_query; ++k) {
            qv[k] = ((const vec *) (q + k*q_s1))[iv];
        }
#pragma unroll
        for (int r = 0; r < DEPTH_MIX_MAX_ROWS; ++r) {
            if (r < n_rows) {
                const vec xv = ((const vec *) (rows.x[r] + t*rows.s[r]))[iv];
#pragma unroll
                for (int c = 0; c < width; ++c) {
                    const float x = depth_mix_get<width>(xv, c);
                    acc[r*(n_query + 1)] += x*x;
#pragma unroll
                    for (int k = 0; k < n_query; ++k) {
                        acc[r*(n_query + 1) + 1 + k] += x*depth_mix_get<width>(qv[k], c);
                    }
                }
            }
        }
    }

    constexpr int n_warps = DEPTH_MIX_BLOCK/warp_size;
    __shared__ float buf[n_acc][n_warps];
    __shared__ float w[DEPTH_MIX_MAX_ROWS][n_query]; // softmax weight / rms of each row
#pragma unroll
    for (int a = 0; a < n_acc; ++a) {
        if (a/(n_query + 1) < n_rows) {
            const float v = warp_reduce_sum<warp_size>(acc[a]);
            if (tid % warp_size == 0) {
                buf[a][tid/warp_size] = v;
            }
        }
    }
    __syncthreads();

    // one warp per accumulator for the second level
    for (int a = tid/warp_size; a < n_rows*(n_query + 1); a += n_warps) {
        float v = (tid % warp_size) < n_warps ? buf[a][tid % warp_size] : 0.0f;
        v = warp_reduce_sum<warp_size>(v);
        if (tid % warp_size == 0) {
            buf[a][0] = v;
        }
    }
    __syncthreads();

    if (tid < n_query) {
        const int k = tid;
        float s[DEPTH_MIX_MAX_ROWS];
        float m = -INFINITY;
        for (int r = 0; r < n_rows; ++r) {
            const float inv = rsqrtf(buf[r*(n_query + 1)][0]/n + eps);
            s[r] = buf[r*(n_query + 1) + 1 + k][0]*inv*scale;
            m = fmaxf(m, s[r]);
        }
        float sum = 0.0f;
        for (int r = 0; r < n_rows; ++r) {
            s[r] = expf(s[r] - m);
            sum += s[r];
        }
        for (int r = 0; r < n_rows; ++r) {
            w[r][k] = s[r]/sum*rsqrtf(buf[r*(n_query + 1)][0]/n + eps);
        }
    }
    __syncthreads();

    ggml_cuda_pdl_lc();
    for (int iv = tid; iv < nv; iv += DEPTH_MIX_BLOCK) {
        float y[n_query][width] = {{0.0f}};
#pragma unroll
        for (int r = 0; r < DEPTH_MIX_MAX_ROWS; ++r) {
            if (r < n_rows) {
                const vec xv = ((const vec *) (rows.x[r] + t*rows.s[r]))[iv];
#pragma unroll
                for (int k = 0; k < n_query; ++k) {
#pragma unroll
                    for (int c = 0; c < width; ++c) {
                        y[k][c] += w[r][k]*depth_mix_get<width>(xv, c);
                    }
                }
            }
        }
#pragma unroll
        for (int k = 0; k < n_query; ++k) {
            float * d = dst + (t*n_query + k)*n + iv*width;
#pragma unroll
            for (int c = 0; c < width; ++c) {
                d[c] = y[k][c];
            }
        }
    }
}

static int depth_mix_n_blocks(const ggml_tensor * op) {
    int n_blocks = 0;
    while (2 + n_blocks < GGML_MAX_SRC && op->src[2 + n_blocks]) {
        n_blocks++;
    }
    return n_blocks;
}

bool ggml_cuda_depth_mix_supported(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) {
        return false;
    }
    ggml_custom_op_params p;
    memcpy(&p, op->op_params, sizeof(p));
    if (p.fun != ggml_depth_mix_cpu) {
        return false;
    }
    const int n_blocks = depth_mix_n_blocks(op);
    if (op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op) || op->src[0]->type != GGML_TYPE_F32 ||
        op->src[0]->nb[0] != sizeof(float) || op->ne[1] < 1 || op->ne[1] > DEPTH_MIX_MAX_QUERY || n_blocks + 1 > DEPTH_MIX_MAX_ROWS) {
        return false;
    }
    for (int r = 0; r <= n_blocks; ++r) {
        const ggml_tensor * x = op->src[1 + r];
        if (x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float)) {
            return false;
        }
    }
    return true;
}

template <int n_query>
static void depth_mix_launch(const ggml_cuda_kernel_launch_params & params, const bool vec4, const depth_mix_rows & rows,
        const int n_rows, const float * q, const int64_t q_s1, const int64_t q_s2, float * d, const int n, const float eps,
        const float scale) {
    if (vec4) {
        ggml_cuda_kernel_launch(depth_mix_f32<n_query, 4>, params, rows, n_rows, q, q_s1, q_s2, d, n, eps, scale);
    } else {
        ggml_cuda_kernel_launch(depth_mix_f32<n_query, 1>, params, rows, n_rows, q, q_s1, q_s2, d, n, eps, scale);
    }
}

void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_custom_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    const ggml_tensor * query    = dst->src[0];
    const int           n_blocks = depth_mix_n_blocks(dst);
    const float         eps      = ggml_depth_mix_eps(p.userdata);
    const int64_t       n        = dst->ne[0];
    const float         scale    = 1.0f/sqrtf((float) n);
    const int64_t       n_query  = dst->ne[1];
    const int64_t       n_tokens = dst->ne[2];

    // src[1] is cur, src[2 + b] block b (oldest first): the rows run newest block first, cur last
    depth_mix_rows rows = {};
    const int n_rows = n_blocks + 1;
    bool vec4 = n % 4 == 0 && (uintptr_t) dst->data % 16 == 0;
    for (int r = 0; r < n_rows; ++r) {
        const ggml_tensor * x = r < n_blocks ? dst->src[2 + (n_blocks - 1 - r)] : dst->src[1];
        rows.x[r] = (const float *) x->data;
        rows.s[r] = x->nb[1]/sizeof(float);
        vec4 = vec4 && (uintptr_t) x->data % 16 == 0 && rows.s[r] % 4 == 0;
    }
    const int64_t q_s1 = query->nb[1]/sizeof(float);
    const int64_t q_s2 = query->ne[2] == 1 ? 0 : query->nb[2]/sizeof(float);
    vec4 = vec4 && (uintptr_t) query->data % 16 == 0 && q_s1 % 4 == 0 && q_s2 % 4 == 0;

    const ggml_cuda_kernel_launch_params params = ggml_cuda_kernel_launch_params(dim3(n_tokens, 1, 1), dim3(DEPTH_MIX_BLOCK, 1, 1), 0, ctx.stream());
    const float * q = (const float *) query->data;
    float       * d = (float *) dst->data;
    switch (n_query) {
        case 1: depth_mix_launch<1>(params, vec4, rows, n_rows, q, q_s1, q_s2, d, (int) n, eps, scale); break;
        case 2: depth_mix_launch<2>(params, vec4, rows, n_rows, q, q_s1, q_s2, d, (int) n, eps, scale); break;
        case 3: depth_mix_launch<3>(params, vec4, rows, n_rows, q, q_s1, q_s2, d, (int) n, eps, scale); break;
        case 4: depth_mix_launch<4>(params, vec4, rows, n_rows, q, q_s1, q_s2, d, (int) n, eps, scale); break;
        default: GGML_ABORT("fatal error");
    }
}
