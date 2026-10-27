#include "hc-fuse.cuh"

// Hyper-connection elementwise chains as single kernels (qwen4exp decode runs ~20 of these small ops per
// layer). Each kernel performs the same float operations in the same order as the unfused ggml ops it
// replaces: scale is `s*x + 0` (scale.cu computes `scale*x + bias`), sigmoid is `1/(1 + expf(-x))` (unary.cu),
// and every product and sum that the unfused graph rounds separately is rounded separately here too
// (__fmul_rn/__fadd_rn stop nvcc contracting them into an FMA). The output is therefore bit-identical.

static __device__ __forceinline__ float hc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// y[i, h, t] = a[i, h, t] + b[i, t] * (s_out*sigmoid(s_in*c[h, t]))
static __global__ void hc_combine_f32(const float * __restrict__ a, const float * __restrict__ b,
                                      const float * __restrict__ c, float * __restrict__ y,
                                      const float s_in, const float s_out,
                                      const int64_t n_embd, const int64_t n_hc, const int64_t n_total) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_total) {
        return;
    }
    const int64_t e = i % n_embd;
    const int64_t r = i / n_embd;        // row = h + n_hc*t
    const int64_t t = r / n_hc;

    const float w = __fmaf_rn(s_out, hc_sigmoid(__fmaf_rn(s_in, c[r], 0.0f)), 0.0f);
    y[i] = __fadd_rn(a[i], __fmul_rn(b[t*n_embd + e], w));
}

// y[i, t] = s * (((x0*sig(g0) + x1*sig(g1)) + x2*sig(g2)) + ...), stream c at offset c*n_embd
static __global__ void hc_mix_tail_f32(const float * __restrict__ x, const float * __restrict__ g,
                                       float * __restrict__ y, const float s,
                                       const int64_t n_embd, const int n_hc, const int64_t n_total) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_total) {
        return;
    }
    const int64_t e = i % n_embd;
    const int64_t t = i / n_embd;
    const float * xt = x + t*n_embd*n_hc + e;
    const float * gt = g + t*n_embd*n_hc + e;

    float acc = __fmul_rn(xt[0], hc_sigmoid(gt[0]));
    for (int ch = 1; ch < n_hc; ++ch) {
        acc = __fadd_rn(acc, __fmul_rn(xt[ch*n_embd], hc_sigmoid(gt[ch*n_embd])));
    }
    y[i] = __fmaf_rn(s, acc, 0.0f);
}

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const ggml_tensor * c = dst->src[2];

    float s_in, s_out;
    memcpy(&s_in,  (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&s_out, (const float *) dst->op_params + 1, sizeof(float));

    const int64_t n_total = ggml_nelements(dst);
    const int bs = 256;
    hc_combine_f32<<<(n_total + bs - 1)/bs, bs, 0, ctx.stream()>>>(
        (const float *) a->data, (const float *) b->data, (const float *) c->data, (float *) dst->data,
        s_in, s_out, a->ne[0], a->ne[1], n_total);
}

void ggml_cuda_op_hc_mix_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * g = dst->src[1];

    const int n_hc = ((const int32_t *) dst->op_params)[0];
    float s;
    memcpy(&s, (const float *) dst->op_params + 1, sizeof(float));

    const int64_t n_total = ggml_nelements(dst);
    const int bs = 256;
    hc_mix_tail_f32<<<(n_total + bs - 1)/bs, bs, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) g->data, (float *) dst->data, s, dst->ne[0], n_hc, n_total);
}

// ggml_hc_mix_fused (enabled by bit 4 of ALPHA_HC_FUSE): the hyper-connection read (norm -> low-rank gate -> gated stream mean,
// plus the inject projection) for decode / verify token counts, as three kernels instead of ~9. Every reduction
// runs in a fixed order that does not depend on how many tokens share the launch, so a token gets the same result
// at T = 1 and inside an MTP verify window.
//   K1  one block per token: per-stream RMS, xn = x*rsqrt(ms+eps)*w_norm, inject = w_inject . xn
//   K2  one block per low-rank row j: lo[j] = silu((w_down[j] . xn) / H)        (16-byte weight loads)
//   K3  one warp per output column d: g_s = w_up[s*N+d] . lo for every stream s, mixed = mean_s xn_s*sigmoid(g_s)

static __device__ __forceinline__ float hcf_bf16(uint32_t b) { return __uint_as_float((b & 0xffffu) << 16); }

template <int H>
static __global__ void hcf_norm_inject(const float * __restrict__ x, const float * __restrict__ wn,
                                       const char * __restrict__ wi, const int64_t wi_nb1, const int wi_bf16,
                                       float * __restrict__ xn, float * __restrict__ dst,
                                       const int N, const float eps) {
    const int t = blockIdx.x;
    const int HD = N*H;
    const float * xt = x + (int64_t) t*HD;
    float * xnt = xn + (int64_t) t*HD;
    __shared__ float red[32][H];
    __shared__ float scl[H];
    float ps[H];
#pragma unroll
    for (int s = 0; s < H; ++s) ps[s] = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
#pragma unroll
        for (int s = 0; s < H; ++s) { const float v = xt[s*N + i]; ps[s] = fmaf(v, v, ps[s]); }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int s = 0; s < H; ++s) {
        float v = ps[s];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) red[wid][s] = v;
    }
    __syncthreads();
    if (wid == 0) {
#pragma unroll
        for (int s = 0; s < H; ++s) {
            float v = lane < nw ? red[lane][s] : 0.0f;
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if (lane == 0) scl[s] = 1.0f/sqrtf(v/(float) N + eps);
        }
    }
    __syncthreads();
    float pi[H];
#pragma unroll
    for (int h = 0; h < H; ++h) pi[h] = 0.0f;
    for (int k = threadIdx.x; k < HD; k += blockDim.x) {
        const int s = k / N;
        const float v = xt[k]*scl[s]*wn[k];
        xnt[k] = v;
        if (wi) {
#pragma unroll
            for (int h = 0; h < H; ++h) {
                const char * row = wi + (int64_t) h*wi_nb1;
                const float w = wi_bf16 ? hcf_bf16(((const uint16_t *) row)[k]) : ((const float *) row)[k];
                pi[h] = fmaf(w, v, pi[h]);
            }
        }
    }
    if (!wi) {
        if (threadIdx.x < H) dst[(int64_t) N*gridDim.x + t*H + threadIdx.x] = 0.0f;
        return;
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < H; ++h) {
        float v = pi[h];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) red[wid][h] = v;
    }
    __syncthreads();
    if (wid == 0) {
#pragma unroll
        for (int h = 0; h < H; ++h) {
            float v = lane < nw ? red[lane][h] : 0.0f;
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if (lane == 0) dst[(int64_t) N*gridDim.x + t*H + h] = v;
        }
    }
}

// 8 consecutive weights (vector v of a row) as floats: BF16 rows are uint4 vectors; Q8_0 rows are 34-byte blocks
// {half d; int8 qs[32]}, 4 vectors per block, only 2-byte aligned
template <bool Q8>
static __device__ __forceinline__ void hcf_load8(const char * __restrict__ row, const int v, float * wf) {
    if constexpr (!Q8) {
        const uint4 q = ((const uint4 *) row)[v];
        wf[0] = hcf_bf16(q.x); wf[1] = hcf_bf16(q.x >> 16);
        wf[2] = hcf_bf16(q.y); wf[3] = hcf_bf16(q.y >> 16);
        wf[4] = hcf_bf16(q.z); wf[5] = hcf_bf16(q.z >> 16);
        wf[6] = hcf_bf16(q.w); wf[7] = hcf_bf16(q.w >> 16);
    } else {
        const char * blk = row + (v >> 2)*34;
        const float d = __half2float(*(const half *) blk);
        const unsigned short * qs = (const unsigned short *) (blk + 2 + (v & 3)*8);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned short u = qs[k];
            wf[2*k + 0] = d*(float) (int8_t) (u & 0xff);
            wf[2*k + 1] = d*(float) (int8_t) (u >> 8);
        }
    }
}

// K2 (two-pass free): one block per low-rank row j (and LR..LR+H-1 = the inject rows). The block reads the raw
// streams x once and accumulates, per token, the per-stream partial dots sum_{k in s} w[k]*x[k]*wn[k] and the
// per-stream sums of squares; the RMS scale is applied to the partials at the end, so no normalised copy of x is
// ever written and the norm kernel is gone. Block 0 also writes the per-(token, stream) scales for K3.
template <int NT, int H, bool Q8>
static __global__ void hcf_down2(const char * __restrict__ wd, const int64_t wd_nb1, const char * __restrict__ wi,
                                 const int64_t wi_nb1, const int wi_bf16,
                                 const float * __restrict__ x, const float * __restrict__ wn,
                                 float * __restrict__ lo, float * __restrict__ scl_out, float * __restrict__ inj_out,
                                 const int N, const int LR, const int T, const float inv_h, const float eps) {
    const int j = blockIdx.x;
    const bool is_inj = j >= LR;
    if (is_inj && !wi) return;
    const int HD = N*H;
    float acc[NT][H], ss[NT][H];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int s = 0; s < H; ++s) { acc[t][s] = 0.0f; ss[t][s] = 0.0f; }
    const int n8 = HD/8;            // 8 elements per vector; a stream holds N/8 vectors (N % 8 == 0)
    const int per_s = N/8;
    for (int v = threadIdx.x; v < n8; v += blockDim.x) {
        const int s = v / per_s;
        float wf[8];
        if (!is_inj) {
            hcf_load8<Q8>(wd + (int64_t) j*wd_nb1, v, wf);
        } else {
            const char * row = wi + (int64_t) (j - LR)*wi_nb1;
            if (wi_bf16) {
                const uint4 q = ((const uint4 *) row)[v];
                wf[0] = hcf_bf16(q.x); wf[1] = hcf_bf16(q.x >> 16);
                wf[2] = hcf_bf16(q.y); wf[3] = hcf_bf16(q.y >> 16);
                wf[4] = hcf_bf16(q.z); wf[5] = hcf_bf16(q.z >> 16);
                wf[6] = hcf_bf16(q.w); wf[7] = hcf_bf16(q.w >> 16);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) wf[e] = ((const float *) row)[v*8 + e];
            }
        }
        const float4 n0 = ((const float4 *) wn)[2*v + 0];
        const float4 n1 = ((const float4 *) wn)[2*v + 1];
        wf[0] *= n0.x; wf[1] *= n0.y; wf[2] *= n0.z; wf[3] *= n0.w;
        wf[4] *= n1.x; wf[5] *= n1.y; wf[6] *= n1.z; wf[7] *= n1.w;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < T) {
                const float4 a = ((const float4 *) (x + (int64_t) t*HD))[2*v + 0];
                const float4 b = ((const float4 *) (x + (int64_t) t*HD))[2*v + 1];
                float d = 0.0f, q2 = 0.0f;
                d = fmaf(wf[0], a.x, d); d = fmaf(wf[1], a.y, d); d = fmaf(wf[2], a.z, d); d = fmaf(wf[3], a.w, d);
                d = fmaf(wf[4], b.x, d); d = fmaf(wf[5], b.y, d); d = fmaf(wf[6], b.z, d); d = fmaf(wf[7], b.w, d);
                q2 = fmaf(a.x, a.x, q2); q2 = fmaf(a.y, a.y, q2); q2 = fmaf(a.z, a.z, q2); q2 = fmaf(a.w, a.w, q2);
                q2 = fmaf(b.x, b.x, q2); q2 = fmaf(b.y, b.y, q2); q2 = fmaf(b.z, b.z, q2); q2 = fmaf(b.w, b.w, q2);
#pragma unroll
                for (int ss_i = 0; ss_i < H; ++ss_i) if (ss_i == s) { acc[t][ss_i] += d; ss[t][ss_i] += q2; }
            }
        }
    }
    __shared__ float red[32][NT][H][2];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int s = 0; s < H; ++s) {
            float a = acc[t][s], b = ss[t][s];
            for (int o = 16; o > 0; o >>= 1) { a += __shfl_xor_sync(0xffffffff, a, o); b += __shfl_xor_sync(0xffffffff, b, o); }
            if (lane == 0) { red[wid][t][s][0] = a; red[wid][t][s][1] = b; }
        }
    __syncthreads();
    if (threadIdx.x < NT*H) {
        const int t = threadIdx.x / H, s = threadIdx.x % H;
        if (t < T) {
            float a = 0.0f, b = 0.0f;
            for (int w = 0; w < nw; ++w) { a += red[w][t][s][0]; b += red[w][t][s][1]; }
            red[0][t][s][0] = a; red[0][t][s][1] = 1.0f/sqrtf(b/(float) N + eps);
        }
    }
    __syncthreads();
    if (threadIdx.x < NT && threadIdx.x < T) {
        const int t = threadIdx.x;
        float v = 0.0f;
#pragma unroll
        for (int s = 0; s < H; ++s) v += red[0][t][s][0]*red[0][t][s][1];
        if (!is_inj) {
            const float a = v*inv_h;
            lo[(int64_t) t*LR + j] = a/(1.0f + expf(-a));
        } else {
            inj_out[(int64_t) t*H + (j - LR)] = v;
        }
        if (j == 0) {
#pragma unroll
            for (int s = 0; s < H; ++s) scl_out[t*H + s] = red[0][t][s][1];
        }
    }
}

// K3: one warp per output column d; xn is rebuilt on the fly from x, the stream scales and w_norm
template <int NT, int H, bool Q8>
static __global__ void hcf_up_mix2(const char * __restrict__ wu, const int64_t wu_nb1, const float * __restrict__ lo,
                                   const float * __restrict__ x, const float * __restrict__ wn, const float * __restrict__ scl,
                                   float * __restrict__ dst, const int N, const int LR, const int T, const float inv_h) {
    extern __shared__ float slo[];   // [T][LR]
    for (int i = threadIdx.x; i < T*LR; i += blockDim.x) slo[i] = lo[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int d = blockIdx.x*(blockDim.x >> 5) + wid;
    if (d >= N) return;
    const int HD = N*H;
    const int n8 = LR/8;
    float g[H][NT];
#pragma unroll
    for (int s = 0; s < H; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t) g[s][t] = 0.0f;
#pragma unroll
    for (int s = 0; s < H; ++s) {
        const char * w = wu + (int64_t) (s*N + d)*wu_nb1;
        for (int v = lane; v < n8; v += 32) {
            float wf[8];
            hcf_load8<Q8>(w, v, wf);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    const float * l = slo + t*LR + v*8;
                    float a = g[s][t];
#pragma unroll
                    for (int e = 0; e < 8; ++e) a = fmaf(wf[e], l[e], a);
                    g[s][t] = a;
                }
            }
        }
    }
#pragma unroll
    for (int s = 0; s < H; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t)
            for (int o = 16; o > 0; o >>= 1) g[s][t] += __shfl_xor_sync(0xffffffff, g[s][t], o);
    if (lane < T && lane < NT) {
        float acc = 0.0f;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t == lane) {
#pragma unroll
                for (int s = 0; s < H; ++s) {
                    const float xn = x[(int64_t) t*HD + s*N + d]*scl[t*H + s]*wn[s*N + d];
                    acc += xn*(1.0f/(1.0f + expf(-g[s][t])));
                }
            }
        }
        dst[(int64_t) lane*N + d] = acc*inv_h;
    }
}

// K2 v4: K-split. Block (jg, ks) computes RW low-rank rows (one warp each) over one K-chunk of KC elements
// (KC divides N, so a chunk lies inside one stream). The chunk of x (all T tokens) and of w_norm is staged in
// shared memory once per block, so L2 traffic is ~weights + (n_jg)*x instead of (LR+4)*x. Partials go to
// part[ks][t][j]; per-chunk sums of squares to pss[ks][t] (written by jg == 0). Every reduction order is fixed
// and independent of T, so a token gets the same bits at T = 1 and inside a verify window.
template <int NT, bool Q8, int RW>
static __global__ void hcf_down4(const char * __restrict__ wd, const int64_t wd_nb1, const char * __restrict__ wi,
                                 const int64_t wi_nb1, const int wi_bf16,
                                 const float * __restrict__ x, const float * __restrict__ wn,
                                 float * __restrict__ part, float * __restrict__ pss,
                                 const int N, const int H, const int LR, const int T, const int KC) {
    extern __shared__ float sm[];          // xs[T][KC], wns[KC]
    float * xs  = sm;
    float * wns = sm + (size_t) T*KC;
    const int jg = blockIdx.x, ks = blockIdx.y, KS = gridDim.y;
    const int HD = N*H;
    const int k0 = ks*KC;
    const int nrows = LR + (wi ? H : 0);
    for (int i = threadIdx.x; i < KC/4; i += blockDim.x) {
        ((float4 *) wns)[i] = ((const float4 *) (wn + k0))[i];
        for (int t = 0; t < T; ++t) ((float4 *) (xs + (size_t) t*KC))[i] = ((const float4 *) (x + (size_t) t*HD + k0))[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (jg == 0) {
        // per-chunk sum of squares, fixed order: thread i sums elements i, i+blockDim, ... then a fixed tree
        __shared__ float red[NT][32];
        for (int t = 0; t < T; ++t) {
            float q = 0.0f;
            for (int i = threadIdx.x; i < KC; i += blockDim.x) { const float v = xs[(size_t) t*KC + i]; q = fmaf(v, v, q); }
            for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffff, q, o);
            if (lane == 0) red[t][w] = q;
        }
        __syncthreads();
        if (threadIdx.x < T) {
            float q = 0.0f;
            for (int i = 0; i < (int) (blockDim.x >> 5); ++i) q += red[threadIdx.x][i];
            pss[ks*T + threadIdx.x] = q;
        }
    }
    const int j = jg*RW + w;
    if (j >= nrows) return;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    const int nv = KC/8;
    for (int v = lane; v < nv; v += 32) {
        float wf[8];
        const int vg = k0/8 + v;   // global vector index within the row
        if (j < LR) {
            hcf_load8<Q8>(wd + (int64_t) j*wd_nb1, vg, wf);
        } else {
            const char * row = wi + (int64_t) (j - LR)*wi_nb1;
            if (wi_bf16) {
                const uint4 q = ((const uint4 *) row)[vg];
                wf[0] = hcf_bf16(q.x); wf[1] = hcf_bf16(q.x >> 16);
                wf[2] = hcf_bf16(q.y); wf[3] = hcf_bf16(q.y >> 16);
                wf[4] = hcf_bf16(q.z); wf[5] = hcf_bf16(q.z >> 16);
                wf[6] = hcf_bf16(q.w); wf[7] = hcf_bf16(q.w >> 16);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) wf[e] = ((const float *) row)[vg*8 + e];
            }
        }
        const float4 n0 = ((const float4 *) wns)[2*v + 0];
        const float4 n1 = ((const float4 *) wns)[2*v + 1];
        wf[0] *= n0.x; wf[1] *= n0.y; wf[2] *= n0.z; wf[3] *= n0.w;
        wf[4] *= n1.x; wf[5] *= n1.y; wf[6] *= n1.z; wf[7] *= n1.w;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < T) {
                const float4 a = ((const float4 *) (xs + (size_t) t*KC))[2*v + 0];
                const float4 b = ((const float4 *) (xs + (size_t) t*KC))[2*v + 1];
                float d = acc[t];
                d = fmaf(wf[0], a.x, d); d = fmaf(wf[1], a.y, d); d = fmaf(wf[2], a.z, d); d = fmaf(wf[3], a.w, d);
                d = fmaf(wf[4], b.x, d); d = fmaf(wf[5], b.y, d); d = fmaf(wf[6], b.z, d); d = fmaf(wf[7], b.w, d);
                acc[t] = d;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
        if (lane == 0 && t < T) part[((size_t) ks*T + t)*nrows + j] = a;
    }
    (void) KS;
}

// K2 v6: hcf_down4 with RR rows per warp. The warp's slice of x (all T tokens) and w_norm is read from shared memory
// once per vector and reused for its RR rows, instead of once per row (shared-memory traffic / RR at T > 1). Per row
// the lane sums the same vectors in the same order with the same operations, then the same shuffle tree: the
// partials are bit-identical to hcf_down4.
template <int NT, bool Q8, int RW, int RR>
static __global__ void hcf_down6(const char * __restrict__ wd, const int64_t wd_nb1, const char * __restrict__ wi,
                                 const int64_t wi_nb1, const int wi_bf16,
                                 const float * __restrict__ x, const float * __restrict__ wn,
                                 float * __restrict__ part, float * __restrict__ pss,
                                 const int N, const int H, const int LR, const int T, const int KC) {
    extern __shared__ float sm[];          // xs[T][KC], wns[KC]
    float * xs  = sm;
    float * wns = sm + (size_t) T*KC;
    const int jg = blockIdx.x, ks = blockIdx.y;
    const int HD = N*H;
    const int k0 = ks*KC;
    const int nrows = LR + (wi ? H : 0);
    for (int i = threadIdx.x; i < KC/4; i += blockDim.x) {
        ((float4 *) wns)[i] = ((const float4 *) (wn + k0))[i];
        for (int t = 0; t < T; ++t) ((float4 *) (xs + (size_t) t*KC))[i] = ((const float4 *) (x + (size_t) t*HD + k0))[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (jg == 0) {
        __shared__ float red[NT][32];
        for (int t = 0; t < T; ++t) {
            float q = 0.0f;
            for (int i = threadIdx.x; i < KC; i += blockDim.x) { const float v = xs[(size_t) t*KC + i]; q = fmaf(v, v, q); }
            for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffff, q, o);
            if (lane == 0) red[t][w] = q;
        }
        __syncthreads();
        if (threadIdx.x < T) {
            float q = 0.0f;
            for (int i = 0; i < (int) (blockDim.x >> 5); ++i) q += red[threadIdx.x][i];
            pss[ks*T + threadIdx.x] = q;
        }
    }
    const int j0 = (jg*RW + w)*RR;
    if (j0 >= nrows) return;
    float acc[RR][NT];
#pragma unroll
    for (int r = 0; r < RR; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
    const int nv = KC/8;
    for (int v = lane; v < nv; v += 32) {
        const int vg = k0/8 + v;
        const float4 n0 = ((const float4 *) wns)[2*v + 0];
        const float4 n1 = ((const float4 *) wns)[2*v + 1];
        float4 xa[NT], xb[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < T) {
                xa[t] = ((const float4 *) (xs + (size_t) t*KC))[2*v + 0];
                xb[t] = ((const float4 *) (xs + (size_t) t*KC))[2*v + 1];
            }
        }
#pragma unroll
        for (int r = 0; r < RR; ++r) {
            const int j = j0 + r;
            if (j >= nrows) break;
            float wf[8];
            if (j < LR) {
                hcf_load8<Q8>(wd + (int64_t) j*wd_nb1, vg, wf);
            } else {
                const char * row = wi + (int64_t) (j - LR)*wi_nb1;
                if (wi_bf16) {
                    const uint4 q = ((const uint4 *) row)[vg];
                    wf[0] = hcf_bf16(q.x); wf[1] = hcf_bf16(q.x >> 16);
                    wf[2] = hcf_bf16(q.y); wf[3] = hcf_bf16(q.y >> 16);
                    wf[4] = hcf_bf16(q.z); wf[5] = hcf_bf16(q.z >> 16);
                    wf[6] = hcf_bf16(q.w); wf[7] = hcf_bf16(q.w >> 16);
                } else {
#pragma unroll
                    for (int e = 0; e < 8; ++e) wf[e] = ((const float *) row)[vg*8 + e];
                }
            }
            wf[0] *= n0.x; wf[1] *= n0.y; wf[2] *= n0.z; wf[3] *= n0.w;
            wf[4] *= n1.x; wf[5] *= n1.y; wf[6] *= n1.z; wf[7] *= n1.w;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    const float4 a = xa[t], b = xb[t];
                    float d = acc[r][t];
                    d = fmaf(wf[0], a.x, d); d = fmaf(wf[1], a.y, d); d = fmaf(wf[2], a.z, d); d = fmaf(wf[3], a.w, d);
                    d = fmaf(wf[4], b.x, d); d = fmaf(wf[5], b.y, d); d = fmaf(wf[6], b.z, d); d = fmaf(wf[7], b.w, d);
                    acc[r][t] = d;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RR; ++r) {
        const int j = j0 + r;
        if (j >= nrows) break;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[r][t];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
            if (lane == 0 && t < T) part[((size_t) ks*T + t)*nrows + j] = a;
        }
    }
}

// finalize: one block per token; thread j combines the KS partials of row j in a fixed order
static __global__ void hcf_down4_fin(const float * __restrict__ part, const float * __restrict__ pss,
                                     float * __restrict__ lo, float * __restrict__ scl_out, float * __restrict__ inj_out,
                                     const int N, const int H, const int LR, const int nrows, const int T, const int KS,
                                     const float inv_h, const float eps) {
    const int t = blockIdx.x;
    const int cps = KS / H;   // chunks per stream
    __shared__ float scl[8];
    if (threadIdx.x < H) {
        float q = 0.0f;
        for (int c = 0; c < cps; ++c) q += pss[(threadIdx.x*cps + c)*T + t];
        const float s = 1.0f/sqrtf(q/(float) N + eps);
        scl[threadIdx.x] = s;
        scl_out[t*H + threadIdx.x] = s;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < nrows; j += blockDim.x) {
        float v = 0.0f;
        for (int s = 0; s < H; ++s) {
            float a = 0.0f;
            for (int c = 0; c < cps; ++c) a += part[((size_t) (s*cps + c)*T + t)*nrows + j];
            v += a*scl[s];
        }
        if (j < LR) {
            const float a = v*inv_h;
            lo[(int64_t) t*LR + j] = a/(1.0f + expf(-a));
        } else {
            inj_out[(int64_t) t*H + (j - LR)] = v;
        }
    }
}

// K3 v3: same math as hcf_up_mix2, but each lane pulls its slice of the low-rank activations out of shared memory
// once (as float4, conflict-light) and reuses it for all H streams, avoiding the 8-way bank conflicts of re-reading
// it per stream with an 8-float lane stride. Accumulation order is unchanged.
template <int NT, int H, bool Q8>
static __global__ void hcf_up_mix3(const char * __restrict__ wu, const int64_t wu_nb1, const float * __restrict__ lo,
                                   const float * __restrict__ x, const float * __restrict__ wn, const float * __restrict__ scl,
                                   float * __restrict__ dst, const int N, const int LR, const int T, const float inv_h) {
    extern __shared__ float slo[];   // [T][LR]
    for (int i = threadIdx.x; i < T*LR; i += blockDim.x) slo[i] = lo[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int d = blockIdx.x*(blockDim.x >> 5) + wid;
    if (d >= N) return;
    const int HD = N*H;
    const int n8 = LR/8;
    constexpr int MAXV = 2;          // vectors per lane held in registers (LR <= 512)
    float l[MAXV][NT][8];
#pragma unroll
    for (int i = 0; i < MAXV; ++i) {
        const int v = lane + 32*i;
        if (v < n8) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    const float4 p = ((const float4 *) (slo + t*LR + v*8))[0];
                    const float4 q = ((const float4 *) (slo + t*LR + v*8))[1];
                    l[i][t][0] = p.x; l[i][t][1] = p.y; l[i][t][2] = p.z; l[i][t][3] = p.w;
                    l[i][t][4] = q.x; l[i][t][5] = q.y; l[i][t][6] = q.z; l[i][t][7] = q.w;
                }
            }
        }
    }
    float g[H][NT];
#pragma unroll
    for (int s = 0; s < H; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t) g[s][t] = 0.0f;
#pragma unroll
    for (int s = 0; s < H; ++s) {
        const char * w = wu + (int64_t) (s*N + d)*wu_nb1;
#pragma unroll
        for (int i = 0; i < MAXV; ++i) {
            const int v = lane + 32*i;
            if (v >= n8) break;
            float wf[8];
            hcf_load8<Q8>(w, v, wf);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    float a = g[s][t];
#pragma unroll
                    for (int e = 0; e < 8; ++e) a = fmaf(wf[e], l[i][t][e], a);
                    g[s][t] = a;
                }
            }
        }
    }
#pragma unroll
    for (int s = 0; s < H; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t)
            for (int o = 16; o > 0; o >>= 1) g[s][t] += __shfl_xor_sync(0xffffffff, g[s][t], o);
    if (lane < T && lane < NT) {
        float acc = 0.0f;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t == lane) {
#pragma unroll
                for (int s = 0; s < H; ++s) {
                    const float xn = x[(int64_t) t*HD + s*N + d]*scl[t*H + s]*wn[s*N + d];
                    acc += xn*(1.0f/(1.0f + expf(-g[s][t])));
                }
            }
        }
        dst[(int64_t) lane*N + d] = acc*inv_h;
    }
}

// up-mix v5 (kernel hcf_up_mix5): up_mix3's arithmetic with the loops swapped -- vector outer, stream inner -- so only one
// vector's activations (NT*8 floats) live in registers instead of MAXV*NT*8, which lowers register pressure and allows
// more resident blocks per SM. Per (stream, token) the accumulation order over vectors is unchanged: bit-identical to v3.
template <int NT, int H, bool Q8>
static __global__ void hcf_up_mix5(const char * __restrict__ wu, const int64_t wu_nb1, const float * __restrict__ lo,
                                   const float * __restrict__ x, const float * __restrict__ wn, const float * __restrict__ scl,
                                   float * __restrict__ dst, const int N, const int LR, const int T, const float inv_h) {
    extern __shared__ float slo[];   // [T][LR]
    for (int i = threadIdx.x; i < T*LR; i += blockDim.x) slo[i] = lo[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int d = blockIdx.x*(blockDim.x >> 5) + wid;
    if (d >= N) return;
    const int HD = N*H;
    const int n8 = LR/8;
    float g[H][NT];
#pragma unroll
    for (int s = 0; s < H; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t) g[s][t] = 0.0f;
    for (int v = lane; v < n8; v += 32) {
        float l[NT][8];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < T) {
                const float4 p = ((const float4 *) (slo + t*LR + v*8))[0];
                const float4 q = ((const float4 *) (slo + t*LR + v*8))[1];
                l[t][0] = p.x; l[t][1] = p.y; l[t][2] = p.z; l[t][3] = p.w;
                l[t][4] = q.x; l[t][5] = q.y; l[t][6] = q.z; l[t][7] = q.w;
            }
        }
#pragma unroll
        for (int s = 0; s < H; ++s) {
            float wf[8];
            hcf_load8<Q8>(wu + (int64_t) (s*N + d)*wu_nb1, v, wf);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    float a = g[s][t];
#pragma unroll
                    for (int e = 0; e < 8; ++e) a = fmaf(wf[e], l[t][e], a);
                    g[s][t] = a;
                }
            }
        }
    }
#pragma unroll
    for (int s = 0; s < H; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t)
            for (int o = 16; o > 0; o >>= 1) g[s][t] += __shfl_xor_sync(0xffffffff, g[s][t], o);
    if (lane < T && lane < NT) {
        float acc = 0.0f;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t == lane) {
#pragma unroll
                for (int s = 0; s < H; ++s) {
                    const float xn = x[(int64_t) t*HD + s*N + d]*scl[t*H + s]*wn[s*N + d];
                    acc += xn*(1.0f/(1.0f + expf(-g[s][t])));
                }
            }
        }
        dst[(int64_t) lane*N + d] = acc*inv_h;
    }
}



template <int NT>
static __global__ void hcf_down(const char * __restrict__ wd, const int64_t wd_nb1, const float * __restrict__ xn,
                                float * __restrict__ lo, const int HD, const int LR, const int T, const float inv_h) {
    const int j = blockIdx.x;
    const uint4 * w = (const uint4 *) (wd + (int64_t) j*wd_nb1);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    const int n8 = HD/8;
    for (int v = threadIdx.x; v < n8; v += blockDim.x) {
        const uint4 q = w[v];
        float wf[8];
        wf[0] = hcf_bf16(q.x); wf[1] = hcf_bf16(q.x >> 16);
        wf[2] = hcf_bf16(q.y); wf[3] = hcf_bf16(q.y >> 16);
        wf[4] = hcf_bf16(q.z); wf[5] = hcf_bf16(q.z >> 16);
        wf[6] = hcf_bf16(q.w); wf[7] = hcf_bf16(q.w >> 16);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < T) {
                const float4 a = ((const float4 *) (xn + (int64_t) t*HD))[2*v + 0];
                const float4 b = ((const float4 *) (xn + (int64_t) t*HD))[2*v + 1];
                float s = acc[t];
                s = fmaf(wf[0], a.x, s); s = fmaf(wf[1], a.y, s); s = fmaf(wf[2], a.z, s); s = fmaf(wf[3], a.w, s);
                s = fmaf(wf[4], b.x, s); s = fmaf(wf[5], b.y, s); s = fmaf(wf[6], b.z, s); s = fmaf(wf[7], b.w, s);
                acc[t] = s;
            }
        }
    }
    __shared__ float red[32][NT];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float v = acc[t];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) red[wid][t] = v;
    }
    __syncthreads();
    if (wid == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = lane < nw ? red[lane][t] : 0.0f;
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if (lane == 0 && t < T) {
                const float a = v*inv_h;
                lo[(int64_t) t*LR + j] = a/(1.0f + expf(-a));
            }
        }
    }
}

template <int NT, int H>
static __global__ void hcf_up_mix(const char * __restrict__ wu, const int64_t wu_nb1, const float * __restrict__ lo,
                                  const float * __restrict__ xn, float * __restrict__ dst,
                                  const int N, const int LR, const int T, const float inv_h) {
    extern __shared__ float slo[];   // [T][LR]
    for (int i = threadIdx.x; i < T*LR; i += blockDim.x) slo[i] = lo[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int d = blockIdx.x*(blockDim.x >> 5) + wid;
    if (d >= N) return;
    const int HD = N*H;
    const int n8 = LR/8;
    float g[H][NT];
#pragma unroll
    for (int s = 0; s < H; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t) g[s][t] = 0.0f;
#pragma unroll
    for (int s = 0; s < H; ++s) {
        const uint4 * w = (const uint4 *) (wu + (int64_t) (s*N + d)*wu_nb1);
        for (int v = lane; v < n8; v += 32) {
            const uint4 q = w[v];
            float wf[8];
            wf[0] = hcf_bf16(q.x); wf[1] = hcf_bf16(q.x >> 16);
            wf[2] = hcf_bf16(q.y); wf[3] = hcf_bf16(q.y >> 16);
            wf[4] = hcf_bf16(q.z); wf[5] = hcf_bf16(q.z >> 16);
            wf[6] = hcf_bf16(q.w); wf[7] = hcf_bf16(q.w >> 16);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    const float * l = slo + t*LR + v*8;
                    float a = g[s][t];
#pragma unroll
                    for (int e = 0; e < 8; ++e) a = fmaf(wf[e], l[e], a);
                    g[s][t] = a;
                }
            }
        }
    }
#pragma unroll
    for (int s = 0; s < H; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t)
            for (int o = 16; o > 0; o >>= 1) g[s][t] += __shfl_xor_sync(0xffffffff, g[s][t], o);
    if (lane < T && lane < NT) {
        float acc = 0.0f;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t == lane) {
#pragma unroll
                for (int s = 0; s < H; ++s) acc += xn[(int64_t) t*HD + s*N + d]*(1.0f/(1.0f + expf(-g[s][t])));
            }
        }
        dst[(int64_t) lane*N + d] = acc*inv_h;
    }
}

bool ggml_cuda_hc_mix_fused_supported(const ggml_tensor * op) {
    const ggml_tensor * x = op->src[0], * wd = op->src[2], * wu = op->src[3], * wi = op->src[4];
    if (ggml_get_op_params_i32(op, 1) == 1) {   // ggml_hc_norm_inject
        return x->type == GGML_TYPE_F32 && x->ne[1] == 4 && (!wi || wi->type == GGML_TYPE_BF16 || wi->type == GGML_TYPE_F32);
    }
    if (x->type != GGML_TYPE_F32 || x->ne[1] != 4 || x->ne[2] > 8) return false;
    if (wd->type != wu->type || (wd->type != GGML_TYPE_BF16 && wd->type != GGML_TYPE_Q8_0)) return false;
    // Q8_0 rows: whole blocks, and 4 eight-wide vectors per block
    if (wd->type == GGML_TYPE_Q8_0 && ((x->ne[0]*x->ne[1]) % QK8_0 != 0 || wd->ne[1] % QK8_0 != 0)) return false;
    if (wi && wi->type != GGML_TYPE_BF16 && wi->type != GGML_TYPE_F32) return false;
    if ((x->ne[0]*x->ne[1]) % 8 != 0 || wd->ne[1] % 8 != 0) return false;
    return true;
}

template <int NT>
static void hcf_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * xn, float * lo) {
    const ggml_tensor * wd = dst->src[2], * wu = dst->src[3];
    const int N = (int) dst->src[0]->ne[0], HD = N*4, LR = (int) wd->ne[1], T = (int) dst->src[0]->ne[2];
    cudaStream_t st = ctx.stream();
    hcf_down<NT><<<LR, 256, 0, st>>>((const char *) wd->data, wd->nb[1], xn, lo, HD, LR, T, 0.25f);
    hcf_up_mix<NT, 4><<<(N + 7)/8, 256, (size_t) T*LR*sizeof(float), st>>>((const char *) wu->data, wu->nb[1], lo, xn,
                                                                             (float *) dst->data, N, LR, T, 0.25f);
}

template <int NT, bool Q8>
static void hcf_launch2(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float eps) {
    const ggml_tensor * x = dst->src[0], * wn = dst->src[1], * wd = dst->src[2], * wu = dst->src[3], * wi = dst->src[4];
    const int N = (int) x->ne[0], LR = (int) wd->ne[1], T = (int) x->ne[2];
    cudaStream_t st = ctx.stream();
    ggml_cuda_pool_alloc<float> lo(ctx.pool(), (size_t) LR*T);
    ggml_cuda_pool_alloc<float> scl(ctx.pool(), (size_t) 4*T);
    float * inj = (float *) dst->data + (size_t) N*T;
    if (!wi) CUDA_CHECK(cudaMemsetAsync(inj, 0, sizeof(float)*4*T, st));
    hcf_down2<NT, 4, Q8><<<LR + 4, 256, 0, st>>>((const char *) wd->data, wd->nb[1], wi ? (const char *) wi->data : nullptr,
        wi ? wi->nb[1] : 0, wi && wi->type == GGML_TYPE_BF16, (const float *) x->data, (const float *) wn->data,
        lo.get(), scl.get(), inj, N, LR, T, 0.25f, eps);
    hcf_up_mix2<NT, 4, Q8><<<(N + 7)/8, 256, (size_t) T*LR*sizeof(float), st>>>((const char *) wu->data, wu->nb[1], lo.get(),
        (const float *) x->data, (const float *) wn->data, scl.get(), (float *) dst->data, N, LR, T, 0.25f);
}

// ALPHA_HC_FUSE_V4 (default on; 0 selects the v2 kernels): K-split read: down4 (x and w_norm staged per K-chunk in
// shared memory) + a fixed-order finalize + up_mix3 (low-rank activations held in registers across the streams).
// Same bits at every T. The v2 K2 re-reads all of x and w_norm from L2 for each of its LR+4 rows, so its L2
// traffic is several times the weight bytes.
static int hcf_v4_chunk(const int N) {
    for (int c = 1; c <= 8; ++c) {
        if (N % c == 0 && (N / c) % 32 == 0 && N / c <= 1280) return N / c;
    }
    return 0;
}
template <int NT>
static void hcf_launch4(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float eps, const int KC) {
    const ggml_tensor * x = dst->src[0], * wn = dst->src[1], * wd = dst->src[2], * wu = dst->src[3], * wi = dst->src[4];
    const int N = (int) x->ne[0], H = 4, LR = (int) wd->ne[1], T = (int) x->ne[2];
    const int nrows = LR + (wi ? H : 0);
    const int KS = H*N / KC;
    constexpr int RW = 8;
    cudaStream_t st = ctx.stream();
    ggml_cuda_pool_alloc<float> lo(ctx.pool(), (size_t) LR*T);
    ggml_cuda_pool_alloc<float> scl(ctx.pool(), (size_t) H*T);
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) KS*T*nrows);
    ggml_cuda_pool_alloc<float> pss(ctx.pool(), (size_t) KS*T);
    float * inj = (float *) dst->data + (size_t) N*T;
    if (!wi) CUDA_CHECK(cudaMemsetAsync(inj, 0, sizeof(float)*H*T, st));
    // ALPHA_HC_DOWN_RR: rows per warp in the down pass (hcf_down6, bit-identical to hcf_down4); 1 = hcf_down4
    static const int rr = []() { const char * e = getenv("ALPHA_HC_DOWN_RR"); return e ? atoi(e) : 2; }();   // default 2
    const size_t dsm = (size_t) (T + 1)*KC*sizeof(float);
    const char * wip = wi ? (const char *) wi->data : nullptr;
    const int64_t wnb = wi ? wi->nb[1] : 0;
    const int wbf = wi && wi->type == GGML_TYPE_BF16;
    if (rr == 2) {
        hcf_down6<NT, true, RW, 2><<<dim3((nrows + RW*2 - 1)/(RW*2), KS), RW*32, dsm, st>>>(
            (const char *) wd->data, wd->nb[1], wip, wnb, wbf, (const float *) x->data, (const float *) wn->data,
            part.get(), pss.get(), N, H, LR, T, KC);
    } else if (rr == 4) {
        hcf_down6<NT, true, RW, 4><<<dim3((nrows + RW*4 - 1)/(RW*4), KS), RW*32, dsm, st>>>(
            (const char *) wd->data, wd->nb[1], wip, wnb, wbf, (const float *) x->data, (const float *) wn->data,
            part.get(), pss.get(), N, H, LR, T, KC);
    } else {
        hcf_down4<NT, true, RW><<<dim3((nrows + RW - 1)/RW, KS), RW*32, dsm, st>>>(
            (const char *) wd->data, wd->nb[1], wip, wnb, wbf, (const float *) x->data, (const float *) wn->data,
            part.get(), pss.get(), N, H, LR, T, KC);
    }
    hcf_down4_fin<<<T, 352, 0, st>>>(part.get(), pss.get(), lo.get(), scl.get(), inj, N, H, LR, nrows, T, KS, 0.25f, eps);
    const size_t usm = (size_t) T*LR*sizeof(float);
    hcf_up_mix5<NT, 4, true><<<(N + 7)/8, 256, usm, st>>>((const char *) wu->data, wu->nb[1], lo.get(),
        (const float *) x->data, (const float *) wn->data, scl.get(), (float *) dst->data, N, LR, T, 0.25f);
}

void ggml_cuda_op_hc_mix_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0], * wn = dst->src[1], * wd = dst->src[2], * wi = dst->src[4];
    if (ggml_get_op_params_i32(dst, 1) == 1) {   // ggml_hc_norm_inject: xn [HD, T] then inject [4, T]
        const int N = (int) x->ne[0], HD = N*4, T = (int) x->ne[2];
        float eps;
        memcpy(&eps, dst->op_params, sizeof(float));
        float * out = (float *) dst->data;
        // the kernel writes inject at dst + N*T + t*4: point dst so that lands right after xn
        hcf_norm_inject<4><<<T, 1024, 0, ctx.stream()>>>((const float *) x->data, (const float *) wn->data,
            wi ? (const char *) wi->data : nullptr, wi ? wi->nb[1] : 0, wi && wi->type == GGML_TYPE_BF16,
            out, out + (int64_t) (HD - N)*T, N, eps);
        return;
    }
    const int N = (int) x->ne[0], HD = N*4, LR = (int) wd->ne[1], T = (int) x->ne[2];
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    const bool q8 = wd->type == GGML_TYPE_Q8_0;
    static const bool v4 = []() { const char * e = getenv("ALPHA_HC_FUSE_V4"); return !e || atoi(e) != 0; }();   // default on
    const int kc = hcf_v4_chunk(N);
    if (v4 && q8 && kc > 0 && LR % 8 == 0 && LR <= 512 && T <= 8) {
        switch (T) {
            case 1:  hcf_launch4<1>(ctx, dst, eps, kc); break;
            case 2:  hcf_launch4<2>(ctx, dst, eps, kc); break;
            case 3:  hcf_launch4<3>(ctx, dst, eps, kc); break;   // an N=2 MTP verify: no padded 4th token
            case 4:  hcf_launch4<4>(ctx, dst, eps, kc); break;
            default: hcf_launch4<8>(ctx, dst, eps, kc); break;
        }
        return;
    }
    if ((N % 8) == 0) {
        if (q8) {
            switch (T) {
                case 1:  hcf_launch2<1, true>(ctx, dst, eps); break;
                case 2:  hcf_launch2<2, true>(ctx, dst, eps); break;
                case 3:
                case 4:  hcf_launch2<4, true>(ctx, dst, eps); break;
                default: hcf_launch2<8, true>(ctx, dst, eps); break;
            }
        } else {
            switch (T) {
                case 1:  hcf_launch2<1, false>(ctx, dst, eps); break;
                case 2:  hcf_launch2<2, false>(ctx, dst, eps); break;
                case 3:
                case 4:  hcf_launch2<4, false>(ctx, dst, eps); break;
                default: hcf_launch2<8, false>(ctx, dst, eps); break;
            }
        }
        return;
    }
    ggml_cuda_pool_alloc<float> xn(ctx.pool(), (size_t) HD*T);
    ggml_cuda_pool_alloc<float> lo(ctx.pool(), (size_t) LR*T);
    hcf_norm_inject<4><<<T, 1024, 0, ctx.stream()>>>((const float *) x->data, (const float *) wn->data,
        wi ? (const char *) wi->data : nullptr, wi ? wi->nb[1] : 0, wi && wi->type == GGML_TYPE_BF16,
        xn.get(), (float *) dst->data, N, eps);
    switch (T) {
        case 1:  hcf_launch<1>(ctx, dst, xn.get(), lo.get()); break;
        case 2:  hcf_launch<2>(ctx, dst, xn.get(), lo.get()); break;
        case 3:
        case 4:  hcf_launch<4>(ctx, dst, xn.get(), lo.get()); break;
        default: hcf_launch<8>(ctx, dst, xn.get(), lo.get()); break;
    }
}
