#include "gated_delta_net.cuh"
#include "gated_delta_net_chunked.cuh"
#include "gated_delta_net_kh.cuh"

#include <cstdlib>

template <int S_v, bool KDA>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state_out_override,   // GGML_GDN_STATE_INPLACE: nullptr = result tail
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = state_out_override ? state_out_override : dst + attn_score_elems;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;
    }

    // Write state back to global memory (transposed layout)
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * warp_size + lane;
        state[col * S_v + i] = s_shard[r];
    }
}

template <bool KDA>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_out_override,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream) {
    // chunked prefill lives in gated_delta_net_chunked.cuh (dispatched from the op entry point)
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    switch (S_v) {
        case 16:
            gated_delta_net_cuda<16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_override, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_cuda<32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_override, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 64: {
            gated_delta_net_cuda<64, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_override, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        case 128: {
            gated_delta_net_cuda<128, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_override, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// GGML_GDN_FUSE_*: one prep launch computes q_hat/k_hat (L2-normalised rows), beta = sigmoid(beta) and
// g = softplus(g + dt[h]) * a[h] into contiguous pool buffers, replacing six elementwise launches per
// DeltaNet layer. One block per (token, seq); a warp per q/k row.
static __global__ void gdn_prep_kernel(
        const float * __restrict__ q, const float * __restrict__ k,
        const float * __restrict__ g, const float * __restrict__ b,
        const float * __restrict__ dt, const float * __restrict__ a,
        float * __restrict__ qh, float * __restrict__ kh, float * __restrict__ gh, float * __restrict__ bh,
        const int S_k, const int H_k, const int H, const int T,
        const int64_t sq1, const int64_t sq2, const int64_t sq3,
        const int64_t sg1, const int64_t sg2, const int64_t sg3,
        const int64_t sb1, const int64_t sb2, const int64_t sb3,
        const float eps, const int flags) {
    const int it = blockIdx.x % T;
    const int ib = blockIdx.x / T;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int nwarp = blockDim.x / WARP_SIZE;
    for (int ih = warp; ih < H_k; ih += nwarp) {
        const float * xq = q + ih*sq1 + it*sq2 + ib*sq3;
        const float * xk = k + ih*sq1 + it*sq2 + ib*sq3;
        float * yq = qh + ((int64_t) (ib*T + it)*H_k + ih) * S_k;
        float * yk = kh + ((int64_t) (ib*T + it)*H_k + ih) * S_k;
        float scq = 1.0f, sck = 1.0f;
        if (flags & GGML_GDN_FUSE_L2NORM_QK) {
            float sq = 0.0f, sk = 0.0f;
            for (int i = lane; i < S_k; i += WARP_SIZE) { sq += xq[i]*xq[i]; sk += xk[i]*xk[i]; }
            sq = warp_reduce_sum(sq);
            sk = warp_reduce_sum(sk);
            scq = 1.0f / fmaxf(sqrtf(sq), eps);
            sck = 1.0f / fmaxf(sqrtf(sk), eps);
        }
        for (int i = lane; i < S_k; i += WARP_SIZE) { yq[i] = xq[i]*scq; yk[i] = xk[i]*sck; }
    }
    for (int ih = threadIdx.x; ih < H; ih += blockDim.x) {
        float gv = g[ih*sg1 + it*sg2 + ib*sg3];
        float bv = b[ih*sb1 + it*sb2 + ib*sb3];
        if (flags & GGML_GDN_FUSE_GATE) {
            const float x = gv + dt[ih];
            gv = ((x > 20.0f) ? x : log1pf(expf(x))) * a[ih];
        }
        if (flags & GGML_GDN_FUSE_SIGMOID_BETA) {
            bv = 1.0f / (1.0f + expf(-bv));
        }
        gh[((int64_t) (ib*T + it))*H + ih] = gv;
        bh[((int64_t) (ib*T + it))*H + ih] = bv;
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;
    const int32_t fuse_flags = ggml_get_op_params_i32(dst, 0);
    // [AI_GDN_PREP_MASK] prep runs for the PREP flags only; STATE_INPLACE alone must not trigger it.
    const int32_t prep_flags = fuse_flags & (GGML_GDN_FUSE_L2NORM_QK | GGML_GDN_FUSE_SIGMOID_BETA |
                                             GGML_GDN_FUSE_GATE);
    // GGML_GDN_STATE_INPLACE: the new state goes straight into the committed cell (src[5]); both kernels
    // are alias-safe (each thread / block round-trips its own state elements through registers or shared)
    float * state_out_override = (fuse_flags & GGML_GDN_STATE_INPLACE) ? (float *) src_state->data : nullptr;
    ggml_cuda_pool_alloc<float> prep_buf(ctx.pool());

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(prep_flags != 0 || ggml_is_contiguous(src_g));      // flagged form: strided views, read by the prep kernel
    GGML_ASSERT(prep_flags != 0 || ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    int64_t sq1 = nbq1 / sizeof(float);
    int64_t sq2 = nbq2 / sizeof(float);
    int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    int64_t sb1 = nbb1 / sizeof(float);
    int64_t sb2 = nbb2 / sizeof(float);
    int64_t sb3 = nbb3 / sizeof(float);
    if (prep_flags != 0) {
        GGML_ASSERT(!kda);
        GGML_ASSERT(src_g->nb[0] == sizeof(float));
        const int64_t S_k = neq0, H_k = neq1;
        const int64_t n_qk = S_k * H_k * n_tokens * n_seqs;
        const int64_t n_h  = H * n_tokens * n_seqs;
        prep_buf.alloc(2 * n_qk + 2 * n_h);
        float * qh = prep_buf.get();
        float * kh = qh + n_qk;
        float * gh = kh + n_qk;
        float * bh = gh + n_h;
        const float * dt_d = dst->src[6] ? (const float *) dst->src[6]->data : nullptr;
        const float * a_d  = dst->src[7] ? (const float *) dst->src[7]->data : nullptr;
        const float   eps  = ggml_get_op_params_f32(dst, 1);
        const int64_t sg1 = src_g->nb[1] / sizeof(float), sg2 = src_g->nb[2] / sizeof(float), sg3 = src_g->nb[3] / sizeof(float);
        const dim3 grid((unsigned) (n_tokens * n_seqs));
        gdn_prep_kernel<<<grid, 128, 0, ctx.stream()>>>(q_d, k_d, g_d, b_d, dt_d, a_d, qh, kh, gh, bh,
            (int) S_k, (int) H_k, (int) H, (int) n_tokens, sq1, sq2, sq3, sg1, sg2, sg3, sb1, sb2, sb3, eps, prep_flags);
        CUDA_CHECK(cudaGetLastError());
        q_d = qh; k_d = kh; g_d = gh; b_d = bh;
        sq1 = S_k; sq2 = S_k * H_k; sq3 = S_k * H_k * n_tokens;
        sb1 = 1;   sb2 = H;         sb3 = H * n_tokens;
    }

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // Chunked (parallel-in-time) prefill for the scalar-gate form. ALPHA_GDN_CHUNK_MIN = minimum
    // n_tokens per sequence to take it (default 32; 0 disables). Decode and MTP verify (1-8 tokens)
    // stay on the sequential kernel, whose per-token latency chain is what the chunked form removes
    // at prefill length. Needs sm_80+ (3xTF32 tensor-core products); see gated_delta_net_chunked.cuh.
    static const int gdn_chunk_min = []() {
        const char * e = getenv("ALPHA_GDN_CHUNK_MIN");
        return e ? atoi(e) : 32;
    }();
    // ALPHA_GDN_KH: the token-serial key-head kernel for prompt-length batches (gated_delta_net_kh.cuh)
    static const int gdn_kh_min = []() {
        const char * e = getenv("ALPHA_GDN_KH");
        return e ? atoi(e) : 64;   // default: prompt batches of >= 64 tokens
    }();
    if (!kda && gdn_kh_min > 0 && n_tokens >= gdn_kh_min &&
        gdn_kh::launch(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_override, S_v, H, neqk1, n_tokens, n_seqs,
                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, rq3, scale, stream)) {
        return;
    }
    if (!kda && gdn_chunk_min > 0 && n_tokens >= gdn_chunk_min && gdn_chunked_supported(S_v, sq2)) {
        if (S_v == 128) {
            gdn_chunked_launch<128>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_override, H, n_tokens, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, ctx);
        } else {
            gdn_chunked_launch<64>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_override, H, n_tokens, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, ctx);
        }
        return;
    }

    if (kda) {
        launch_gated_delta_net<true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_override,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    } else {
        launch_gated_delta_net<false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_override,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    }
}
