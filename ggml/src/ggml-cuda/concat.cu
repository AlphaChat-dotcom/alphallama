#include "concat.cuh"

// contiguous kernels
static __global__ void concat_f32_dim0(const float * x, const float * y, float * dst, const int ne0, const int ne00) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim1(const float * x, const float * y, float * dst, const int ne0, const int ne01) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.y < (unsigned)ne01) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            (blockIdx.y - ne01) * ne0 +
            blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim2(const float * x, const float * y, float * dst, const int ne0, const int ne02) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.z < (unsigned)ne02) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 *  gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f32_cuda(const float * x, const float * y, float * dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2);
    if (dim == 0) {
        concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00);
        return;
    }
    if (dim == 1) {
        concat_f32_dim1<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne01);
        return;
    }
    concat_f32_dim2<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne02);
}

// non-contiguous kernel (slow)
template <int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_f32_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3){
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const float * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const float *) (src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            } else if constexpr (dim == 1) {
                x = (const float *) (src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            } else if constexpr (dim == 2) {
                x = (const float *) (src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            } else if constexpr (dim == 3) {
                x = (const float *) (src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        float * y = (float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


// dim-0 concat of short rows (e.g. the recurrent conv input concat(state[k, C], x^T[T, C]) of an MTP verify,
// where the fused concat_shift does not apply because the fold owns the state write): the generic non-contiguous
// kernel spends one block per row on k+T <= 32 elements. One thread per row.
static __global__ void concat_f32_dim0_rows(
        const char * __restrict__ s0, const char * __restrict__ s1, char * __restrict__ dst,
        const int ne00, const int ne0, const int ne1, const int nrows,
        const int64_t nb00, const int64_t nb01, const int64_t nb02,
        const int64_t nb10, const int64_t nb11, const int64_t nb12,
        const int64_t nb0,  const int64_t nb1,  const int64_t nb2) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) {
        return;
    }
    const int i1 = r % ne1, i2 = r / ne1;
    const char * a = s0 + i1*nb01 + i2*nb02;
    const char * b = s1 + i1*nb11 + i2*nb12;
    char       * d = dst + i1*nb1 + i2*nb2;
    for (int i0 = 0; i0 < ne0; i0++) {
        const float v = i0 < ne00 ? *(const float *) (a + i0*nb00) : *(const float *) (b + (i0 - ne00)*nb10);
        *(float *) (d + i0*nb0) = v;
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    // F32, or I32 copied as 4-byte words (no arithmetic touches the values, so the bits are preserved)
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == src0->type && dst->type == src0->type);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const float * src0_d = (const float *)src0->data;
        const float * src1_d = (const float *)src1->data;

        float * dst_d = (float *)dst->data;

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_f32_cuda(
                        src0_d + i3 * (src0->nb[3] / 4),
                        src1_d + i3 * (src1->nb[3] / 4),
                        dst_d + i3 * ( dst->nb[3] / 4),
                        src0->ne[0], src0->ne[1], src0->ne[2],
                        dst->ne[0],  dst->ne[1],  dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            CUDA_CHECK(cudaMemcpyAsync(dst_d,           src0_d, size0, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dst_d + size0/4, src1_d, size1, cudaMemcpyDeviceToDevice, stream));
        }
    } else if (dim == 0 && dst->ne[0] <= 32 && dst->ne[3] == 1 && src0->ne[3] == 1 && src1->ne[3] == 1 &&
               dst->ne[1]*dst->ne[2] <= INT32_MAX) {
        const int nrows = (int) (dst->ne[1]*dst->ne[2]);
        concat_f32_dim0_rows<<<(nrows + 255)/256, 256, 0, stream>>>(
            (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
            (int) src0->ne[0], (int) dst->ne[0], (int) dst->ne[1], nrows,
            src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[0], src1->nb[1], src1->nb[2],
            dst->nb[0], dst->nb[1], dst->nb[2]);
    } else {
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_f32_non_cont<dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}

// Recurrent conv-state roll, one launch instead of CONCAT + CPY. At decode the graph builds
// conv_input = concat(state[k, C], x^T[T, C]) and copies its last k columns back into the state buffer; the
// generic concat kernel would spend one 256-thread block per channel on 4 elements.
//   out[t, c, b]            = t < k ? s[t, c, b] : x[t - k, c, b]     out contiguous [k+T, C, B]
//   st[j + k*(c + C*b)]      = out[T + j, c, b], j < k                  the CPY destination, contiguous
// One thread per (channel, batch); it reads its k state values before writing them, so an in-place roll
// (state buffer == s) is safe too.
static __global__ void concat_shift_f32(
        const char * __restrict__ s, const char * __restrict__ x, float * __restrict__ out, float * __restrict__ st,
        const int k, const int T, const int C,
        const int64_t snb1, const int64_t snb2,
        const int64_t xnb0, const int64_t xnb1, const int64_t xnb2) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= C) {
        return;
    }
    const float * sp = (const float *) (s + (int64_t) c * snb1 + (int64_t) b * snb2);
    float * op  = out + ((int64_t) b * C + c) * (k + T);
    float * stp = st  + ((int64_t) b * C + c) * k;
    float sv[8];
    for (int t = 0; t < k; t++) { sv[t] = sp[t]; }
    for (int t = 0; t < k; t++) { op[t] = sv[t]; }
    for (int t = 0; t < T; t++) {
        op[k + t] = *(const float *) (x + (int64_t) t * xnb0 + (int64_t) c * xnb1 + (int64_t) b * xnb2);
    }
    for (int j = 0; j < k; j++) { stp[j] = op[T + j]; }
}

void ggml_cuda_op_concat_shift(ggml_backend_cuda_context & ctx, ggml_tensor * concat, ggml_tensor * cpy) {
    const ggml_tensor * s = concat->src[0];
    const ggml_tensor * x = concat->src[1];
    const int k = (int) s->ne[0];
    const int T = (int) x->ne[0];
    const int C = (int) s->ne[1];
    const int B = (int) s->ne[2];
    GGML_ASSERT(k >= 1 && k <= 8 && T >= 1);
    GGML_ASSERT(ggml_is_contiguous(concat) && s->nb[0] == sizeof(float));
    const dim3 grid((C + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE, B, 1);
    concat_shift_f32<<<grid, CUDA_CONCAT_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const char *) s->data, (const char *) x->data, (float *) concat->data, (float *) cpy->data,
        k, T, C, (int64_t) s->nb[1], (int64_t) s->nb[2], (int64_t) x->nb[0], (int64_t) x->nb[1], (int64_t) x->nb[2]);
}
