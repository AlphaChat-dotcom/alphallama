#include "loop-ctrl.cuh"

// the block size mul_mat_vec_f picks for a row of ncols floats (mmvf.cu), so that the dot products reduce alike
static int loop_ctrl_block_size(const int64_t ncols, const int warp_size) {
    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    int64_t max_block_size  = 256;
    if (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    return (int) block_size_best;
}

// gate = D x [cur_norm; msg_norm]: one block per rank row, the reduction of mul_mat_vec_f for F32 with the same block size
template <int ncols_dst, int block_size>
static __global__ void loop_ctrl_gate(const float * __restrict__ D, const float * __restrict__ cn, const float * __restrict__ mn,
        float * __restrict__ gate, const int ncols2, const int half2, const int64_t s_cn, const int64_t s_mn, const int rank) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    const float2 * x2 = (const float2 *) (D + int64_t(row)*2*ncols2);

    __shared__ float buf_iw[warp_size];
    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = {0.0f};

    // unrolled so that the loads run ahead; the sums keep their order
#pragma unroll 4
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            // the concatenation: cur_norm, then msg_norm
            const float2 tmpy = col2 < half2 ? ((const float2 *) (cn + j*s_cn))[col2] : ((const float2 *) (mn + j*s_mn))[col2 - half2];
            ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
        }
    }

    ggml_cuda_pdl_lc();
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if (block_size > warp_size) {
            buf_iw[tid/warp_size] = sumf[j];
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
            }
            __syncthreads();
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    gate[int64_t(tid)*rank + row] = sumf[tid];
}

// one warp per output row: forget and gain as mul_mat_vec_f reduces a row of rank <= 2*warp_size floats, then
// y = (cur - tanh(forget)*cur) + tanh(gain)*msg with each product and sum rounded separately
template <int ncols_dst>
static __global__ void loop_ctrl_out(const float * __restrict__ F, const float * __restrict__ G, const float * __restrict__ gate,
        const float * cur, const float * msg, float * dst, const int rank,
        const int64_t s_cur, const int64_t s_msg, const int64_t s_dst) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    const float2 * f2 = (const float2 *) (F + int64_t(row)*rank);
    const float2 * g2 = (const float2 *) (G + int64_t(row)*rank);

    float sumf[ncols_dst] = {0.0f};
    float sumg[ncols_dst] = {0.0f};

    for (int col2 = tid; col2 < rank/2; col2 += warp_size) {
        const float2 tmpf = f2[col2];
        const float2 tmpg = g2[col2];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float2 tmpy = ((const float2 *) (gate + int64_t(j)*rank))[col2];
            ggml_cuda_mad(sumf[j], tmpf.x, tmpy.x);
            ggml_cuda_mad(sumf[j], tmpf.y, tmpy.y);
            ggml_cuda_mad(sumg[j], tmpg.x, tmpy.x);
            ggml_cuda_mad(sumg[j], tmpg.y, tmpy.y);
        }
    }

    ggml_cuda_pdl_lc();
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
        sumg[j] = warp_reduce_sum<warp_size>(sumg[j]);
    }

    if (tid >= ncols_dst) {
        return;
    }

    const float forget = tanhf(sumf[tid]);
    const float gain   = tanhf(sumg[tid]);
    const float c      = cur[tid*s_cur + row];
    const float m      = msg[tid*s_msg + row];

    dst[tid*s_dst + row] = __fadd_rn(__fsub_rn(c, __fmul_rn(forget, c)), __fmul_rn(gain, m));
}

template <int ncols_dst>
static void loop_ctrl_launch(ggml_backend_cuda_context & ctx, const float * D, const float * F, const float * G,
        const float * cn, const float * mn, const float * cur, const float * msg, float * dst, float * gate,
        const int64_t n, const int rank, const int64_t s_cn, const int64_t s_mn, const int64_t s_cur, const int64_t s_msg,
        const int64_t s_dst) {
    const int warp_size  = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int block_size = loop_ctrl_block_size(2*n, warp_size);
    cudaStream_t stream  = ctx.stream();

    const ggml_cuda_kernel_launch_params gate_params = ggml_cuda_kernel_launch_params(dim3(rank, 1, 1), dim3(block_size, 1, 1), 0, stream);
    const int ncols2 = (int) n;      // 2*n floats
    const int half2  = (int) (n/2);  // cur_norm, in float2
    switch (block_size) {
        case  32: ggml_cuda_kernel_launch(loop_ctrl_gate<ncols_dst,  32>, gate_params, D, cn, mn, gate, ncols2, half2, s_cn, s_mn, rank); break;
        case  64: ggml_cuda_kernel_launch(loop_ctrl_gate<ncols_dst,  64>, gate_params, D, cn, mn, gate, ncols2, half2, s_cn, s_mn, rank); break;
        case  96: ggml_cuda_kernel_launch(loop_ctrl_gate<ncols_dst,  96>, gate_params, D, cn, mn, gate, ncols2, half2, s_cn, s_mn, rank); break;
        case 128: ggml_cuda_kernel_launch(loop_ctrl_gate<ncols_dst, 128>, gate_params, D, cn, mn, gate, ncols2, half2, s_cn, s_mn, rank); break;
        case 160: ggml_cuda_kernel_launch(loop_ctrl_gate<ncols_dst, 160>, gate_params, D, cn, mn, gate, ncols2, half2, s_cn, s_mn, rank); break;
        case 192: ggml_cuda_kernel_launch(loop_ctrl_gate<ncols_dst, 192>, gate_params, D, cn, mn, gate, ncols2, half2, s_cn, s_mn, rank); break;
        case 224: ggml_cuda_kernel_launch(loop_ctrl_gate<ncols_dst, 224>, gate_params, D, cn, mn, gate, ncols2, half2, s_cn, s_mn, rank); break;
        case 256: ggml_cuda_kernel_launch(loop_ctrl_gate<ncols_dst, 256>, gate_params, D, cn, mn, gate, ncols2, half2, s_cn, s_mn, rank); break;
        default: GGML_ABORT("fatal error");
    }

    // several output rows per block, as mul_mat_vec_f packs short rows
    int rows_per_block = 1;
    for (const int r : {8, 4, 2}) {
        if (n % r == 0) {
            rows_per_block = r;
            break;
        }
    }
    const ggml_cuda_kernel_launch_params out_params = ggml_cuda_kernel_launch_params(
            dim3(n/rows_per_block, 1, 1), dim3(warp_size, rows_per_block, 1), 0, stream);
    ggml_cuda_kernel_launch(loop_ctrl_out<ncols_dst>, out_params, F, G, (const float *) gate, cur, msg, dst, rank, s_cur, s_msg, s_dst);
}

void ggml_cuda_op_loop_ctrl_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * cur, const ggml_tensor * cur_norm, const ggml_tensor * msg, const ggml_tensor * msg_norm,
        const ggml_tensor * D, const ggml_tensor * F, const ggml_tensor * G, ggml_tensor * dst) {
    const int64_t n        = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const int     rank     = (int) D->ne[1];

    GGML_ASSERT(n_tokens >= 1 && n_tokens <= LOOP_CTRL_MAX_TOKENS);

    ggml_cuda_pool_alloc<float> gate(ctx.pool(), rank*n_tokens);

    const float * Dd  = (const float *) D->data;
    const float * Fd  = (const float *) F->data;
    const float * Gd  = (const float *) G->data;
    const float * cnd = (const float *) cur_norm->data;
    const float * mnd = (const float *) msg_norm->data;
    const float * cd  = (const float *) cur->data;
    const float * md  = (const float *) msg->data;
    float       * dd  = (float *) dst->data;

    const int64_t s_cn  = cur_norm->nb[1]/sizeof(float);
    const int64_t s_mn  = msg_norm->nb[1]/sizeof(float);
    const int64_t s_cur = cur->nb[1]/sizeof(float);
    const int64_t s_msg = msg->nb[1]/sizeof(float);
    const int64_t s_dst = dst->nb[1]/sizeof(float);

#define LOOP_CTRL_CASE(T) case T: loop_ctrl_launch<T>(ctx, Dd, Fd, Gd, cnd, mnd, cd, md, dd, gate.get(), n, rank, s_cn, s_mn, s_cur, s_msg, s_dst); break;
    switch (n_tokens) {
        LOOP_CTRL_CASE(1)
        LOOP_CTRL_CASE(2)
        LOOP_CTRL_CASE(3)
        LOOP_CTRL_CASE(4)
        LOOP_CTRL_CASE(5)
        LOOP_CTRL_CASE(6)
        LOOP_CTRL_CASE(7)
        LOOP_CTRL_CASE(8)
        default: GGML_ABORT("fatal error");
    }
#undef LOOP_CTRL_CASE
}
