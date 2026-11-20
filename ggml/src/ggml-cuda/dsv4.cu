#include "common.cuh"
#include <cstring>
#include <cuda_fp16.h>
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

// Fused mixer pre side (ALPHA_HC_PRE_FUSED, default 1), two kernels.
//   x     [n_embd, n_hc, n_tok] contiguous (the residual streams);  D = n_embd * n_hc
//   fn    n_mix rows of D Q8_0 weights (blocks of 32: f16 scale + 32 int8)
//   mixes [n_mix, n_tok]: fn . rms_norm(x)   (the MUL_MAT node's own output buffer)
//   split [n_mix, n_tok]: pre (n_hc) | post (n_hc) | comb (n_hc * n_hc), as dsv4_hc_split_sinkhorn_k writes it
//   out   [n_embd, n_tok]: sum over streams of pre[h] * x[:, h]
// The mix logits use the float streams (the separate MUL_MAT quantises them to 8 bits first), so they differ from the
// unfused path in the last digits.
// Kernel 1: one block per (logit row, token); thread b owns weight block b (32 weights) and the same 32 stream values.
static __global__ void dsv4_hc_mix_k(const float * __restrict__ x, const char * __restrict__ fn, const size_t fn_row,
        float * __restrict__ mixes, const int D, const float rms_eps, const size_t mix_stride) {
    const int r = blockIdx.x, t = blockIdx.y;
    const float * xt = x + (size_t) t * D;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    __shared__ float sh_ss[32], sh_dot[32];
    float ss = 0.0f, dot = 0.0f;
    const int n_blk = D / 32;
    for (int bk = tid; bk < n_blk; bk += blockDim.x) {
        const char * blk = fn + (size_t) r * fn_row + (size_t) bk * 34;
        const float d = __half2float(*(const __half *) blk);
        const int8_t * qs = (const int8_t *) (blk + 2);
        const float * xv = xt + bk * 32;
        float sq = 0.0f, sd = 0.0f;
        for (int i = 0; i < 32; i++) { const float v = xv[i]; sq += v * v; sd += (float) qs[i] * v; }
        ss += sq; dot += d * sd;
    }
    ss = warp_reduce_sum(ss); dot = warp_reduce_sum(dot);
    if (lane == 0) { sh_ss[warp] = ss; sh_dot[warp] = dot; }
    __syncthreads();
    if (tid == 0) {
        float tss = 0.0f, tdot = 0.0f;
        for (int w = 0; w < nw; w++) { tss += sh_ss[w]; tdot += sh_dot[w]; }
        mixes[(size_t) t * mix_stride + r] = tdot * rsqrtf(tss / D + rms_eps);
    }
}
// Kernel 2: one block per (256 outputs, token). Every block derives pre from the logits; block 0 also writes the split
// (its lanes 0..15 run the Sinkhorn iterations, lane = sh + dh * n_hc as in dsv4_hc_split_sinkhorn_k).
static __global__ void dsv4_hc_split_wsum_k(const float * __restrict__ x, const float * __restrict__ mixes,
        const float * __restrict__ scale, const float * __restrict__ base, float * __restrict__ split, float * __restrict__ out,
        const int n_embd, const int sinkhorn_iters, const float eps,
        const size_t mix_stride, const size_t split_stride, const size_t out_stride) {
    constexpr int n_hc = 4;
    const int t = blockIdx.y;
    const float * xt = x + (size_t) t * n_embd * n_hc;
    const float * m = mixes + (size_t) t * mix_stride;
    const int tid = threadIdx.x;
    const float ps = scale[0];
    float pre[4];
    for (int h = 0; h < n_hc; h++) pre[h] = 1.0f / (1.0f + expf(-(m[h] * ps + base[h]))) + eps;
    const int d = blockIdx.x * blockDim.x + tid;
    if (d < n_embd) {
        float a = 0.0f;
        a += xt[d] * pre[0];
        a += xt[n_embd + d] * pre[1];
        a += xt[2 * n_embd + d] * pre[2];
        a += xt[3 * n_embd + d] * pre[3];
        out[(size_t) t * out_stride + d] = a;
    }
    if (blockIdx.x != 0 || tid >= 16) return;
    __shared__ float sh_c[16];
    const float pos = scale[1], cs = scale[2];
    const int sh = tid & 3, dh = tid >> 2;
    float * dst = split + (size_t) t * split_stride;
    if (tid < n_hc) {
        dst[tid] = pre[tid];
        dst[n_hc + tid] = 2.0f / (1.0f + expf(-(m[n_hc + tid] * pos + base[n_hc + tid])));
    }
    float row[4];
    float mx = -INFINITY;
    for (int k = 0; k < n_hc; k++) {
        const int off = 2 * n_hc + k + dh * n_hc;
        row[k] = m[off] * cs + base[off];
        mx = fmaxf(mx, row[k]);
    }
    float s = 0.0f;
    for (int k = 0; k < n_hc; k++) { row[k] = expf(row[k] - mx); s += row[k]; }
    float c = row[sh] / s + eps;
    sh_c[tid] = c;
    __syncwarp(0xffffu);
    {
        float cs2 = 0.0f; for (int k = 0; k < n_hc; k++) cs2 += sh_c[sh + k * n_hc];
        __syncwarp(0xffffu);
        c *= 1.0f / (cs2 + eps);
        sh_c[tid] = c;
        __syncwarp(0xffffu);
    }
    for (int it = 0; it < sinkhorn_iters - 1; ++it) {
        float rs = 0.0f; for (int k = 0; k < n_hc; k++) rs += sh_c[k + dh * n_hc];
        __syncwarp(0xffffu);
        c *= 1.0f / (rs + eps);
        sh_c[tid] = c;
        __syncwarp(0xffffu);
        float cs2 = 0.0f; for (int k = 0; k < n_hc; k++) cs2 += sh_c[sh + k * n_hc];
        __syncwarp(0xffffu);
        c *= 1.0f / (cs2 + eps);
        sh_c[tid] = c;
        __syncwarp(0xffffu);
    }
    dst[2 * n_hc + tid] = c;
}

// Tries to run nodes i.. (RMS_NORM, MUL_MAT, DSV4_HC_SPLIT_SINKHORN, views, DSV4_HC_WEIGHTED_SUM) as the fused kernel.
// Returns the index of the last node it covered, or -1.
int ggml_cuda_dsv4_hc_pre_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool on = [] { const char * e = getenv("ALPHA_HC_PRE_FUSED"); return !e || atoi(e) != 0; }();
    if (!on || i + 3 >= cgraph->n_nodes) return -1;
    const ggml_tensor * rn = cgraph->nodes[i];
    const ggml_tensor * mm = cgraph->nodes[i + 1];
    const ggml_tensor * sp = cgraph->nodes[i + 2];
    if (rn->op != GGML_OP_RMS_NORM || mm->op != GGML_OP_MUL_MAT || sp->op != GGML_OP_DSV4_HC_SPLIT_SINKHORN) return -1;
    if (mm->src[1] != rn || sp->src[0] != mm || !mm->src[0] || mm->src[0]->type != GGML_TYPE_Q8_0) return -1;
    const ggml_tensor * flat = rn->src[0];
    const ggml_tensor * fnw = mm->src[0];
    const int n_hc = ((const int32_t *) sp->op_params)[0];
    const int si = ((const int32_t *) sp->op_params)[1];
    float eps; memcpy(&eps, (const int32_t *) sp->op_params + 2, sizeof(float));
    float rms_eps; memcpy(&rms_eps, rn->op_params, sizeof(float));
    const int64_t D = flat->ne[0], nt = flat->ne[1], n_mix = fnw->ne[1];
    if (n_hc != 4 || n_mix != (2 + n_hc) * n_hc || fnw->ne[0] != D || D % (32 * n_hc) != 0 || nt < 1 || nt > 64) return -1;
    if (flat->type != GGML_TYPE_F32 || !ggml_is_contiguous(flat) || sp->type != GGML_TYPE_F32 || sp->nb[0] != sizeof(float)) return -1;
    if (flat->ne[2] != 1 || flat->ne[3] != 1 || sp->ne[0] != n_mix || sp->ne[1] != nt) return -1;
    if (!sp->src[1] || !sp->src[2] || sp->src[1]->type != GGML_TYPE_F32 || sp->src[2]->type != GGML_TYPE_F32) return -1;
    // views of the split, then the weighted sum over the same streams
    int j = i + 3;
    while (j < cgraph->n_nodes && j < i + 12 && (cgraph->nodes[j]->op == GGML_OP_VIEW || cgraph->nodes[j]->op == GGML_OP_RESHAPE)) j++;
    if (j >= cgraph->n_nodes) return -1;
    const ggml_tensor * ws = cgraph->nodes[j];
    if (ws->op != GGML_OP_DSV4_HC_WEIGHTED_SUM || !ws->src[0] || !ws->src[1]) return -1;
    const ggml_tensor * x = ws->src[0];
    const ggml_tensor * pre = ws->src[1];
    if (x->data != flat->data || !ggml_is_contiguous(x) || x->ne[1] != n_hc || x->ne[0] * n_hc != D || x->ne[2] != nt) return -1;
    if (pre->view_src != sp || pre->data != sp->data || ws->type != GGML_TYPE_F32 || ws->nb[0] != sizeof(float) || ws->ne[0] != x->ne[0]) return -1;
    if (mm->type != GGML_TYPE_F32 || mm->nb[0] != sizeof(float)) return -1;
    { static bool said = false; if (!said) { said = true; fprintf(stderr, "[hc-pre] mixer pre side fused: %d logits, %d tokens\n", (int) n_mix, (int) nt); } }
    dsv4_hc_mix_k<<<dim3((unsigned) n_mix, (unsigned) nt), 512, 0, ctx.stream()>>>((const float *) flat->data, (const char *) fnw->data,
            fnw->nb[1], (float *) mm->data, (int) D, rms_eps, mm->nb[1] / sizeof(float));
    dsv4_hc_split_wsum_k<<<dim3((unsigned) ((x->ne[0] + 255) / 256), (unsigned) nt), 256, 0, ctx.stream()>>>((const float *) flat->data,
            (const float *) mm->data, (const float *) sp->src[1]->data, (const float *) sp->src[2]->data, (float *) sp->data,
            (float *) ws->data, (int) x->ne[0], si, eps, mm->nb[1] / sizeof(float), sp->nb[1] / sizeof(float), ws->nb[1] / sizeof(float));
    return j;
}

// MoE router tail (ALPHA_ROUTER_FUSED, default 1): one block per token, n_expert <= 512.
//   logits [n_exp, n_tok] -> probs = sigmoid(logits) (written: the weights are gathered from it)
//   order  [n_exp, n_tok] int32: experts by descending probs + bias (ties: lower index first)
//   w      [1, k, n_tok]: probs of the first k, divided by max(their sum, sum_min), times w_scale
static __global__ void moe_router_tail_k(const float * __restrict__ logits, const float * __restrict__ bias, float * __restrict__ probs,
        int32_t * __restrict__ order, float * __restrict__ w, const int n_exp, const int k_used, const float sum_min, const float w_scale,
        const size_t logits_stride, const size_t probs_stride, const size_t order_stride, const size_t w_k_stride, const size_t w_t_stride) {
    __shared__ float sh_v[512];
    __shared__ float sh_w[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    float p = 0.0f;
    if (tid < n_exp) {
        p = 1.0f / (1.0f + expf(-logits[(size_t) t * logits_stride + tid]));
        probs[(size_t) t * probs_stride + tid] = p;
        sh_v[tid] = p + bias[tid];
    }
    __syncthreads();
    if (tid < n_exp) {
        const float v = sh_v[tid];
        int rank = 0;
        for (int i = 0; i < n_exp; i++) {
            const float u = sh_v[i];
            rank += (u > v || (u == v && i < tid)) ? 1 : 0;
        }
        order[(size_t) t * order_stride + rank] = tid;
        if (rank < k_used) sh_w[rank] = p;
    }
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int k = 0; k < k_used; k++) s += sh_w[k];
        s = fmaxf(s, sum_min);
        for (int k = 0; k < k_used; k++) w[(size_t) t * w_t_stride + (size_t) k * w_k_stride] = (sh_w[k] / s) * w_scale;
    }
}

void ggml_cuda_dre_node_done(ggml_backend_cuda_context & ctx, const ggml_tensor * node);   // dre.cu: early doorbell

// Tries to run nodes i.. (sigmoid, + bias, argsort, get_rows, sum_rows, clamp, div, scale; views between) as one kernel.
// Returns the index of the last node covered, or -1.
int ggml_cuda_moe_router_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool on = [] { const char * e = getenv("ALPHA_ROUTER_FUSED"); return !e || atoi(e) != 0; }();
    if (!on || i + 7 >= cgraph->n_nodes) return -1;
    const ggml_tensor * sg = cgraph->nodes[i];
    if (sg->op != GGML_OP_UNARY || (ggml_unary_op) ((const int32_t *) sg->op_params)[0] != GGML_UNARY_OP_SIGMOID) return -1;
    const ggml_tensor * lg = sg->src[0];
    if (!lg || lg->op != GGML_OP_MUL_MAT || !lg->src[0] || !strstr(lg->src[0]->name, "ffn_gate_inp")) return -1;
    const ggml_tensor * ad = nullptr, * as = nullptr;
    const ggml_tensor * gr = nullptr, * sr = nullptr, * cl = nullptr, * dv = nullptr, * scl = nullptr;
    int j = i + 1;
    for (; j < cgraph->n_nodes && j < i + 24 && !scl; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        switch (n->op) {
            case GGML_OP_VIEW: case GGML_OP_RESHAPE: break;
            case GGML_OP_ADD:      if (ad || n->src[0] != sg || !n->src[1]) return -1; ad = n; break;
            case GGML_OP_ARGSORT:  if (!ad || as || n->src[0] != ad) return -1; as = n; break;
            case GGML_OP_GET_ROWS: if (!as || gr) return -1; gr = n; break;
            case GGML_OP_SUM_ROWS: if (!gr || sr) return -1; sr = n; break;
            case GGML_OP_CLAMP:    if (!sr || cl || n->src[0] != sr) return -1; cl = n; break;
            case GGML_OP_DIV:      if (!cl || dv || n->src[1] != cl) return -1; dv = n; break;
            case GGML_OP_SCALE:    if (!dv) return -1; scl = n; break;
            default: return -1;
        }
    }
    if (!scl) return -1;
    j--;   // index of the SCALE node
    if ((ggml_sort_order) ((const int32_t *) as->op_params)[0] != GGML_SORT_ORDER_DESC) return -1;
    const int64_t n_exp = sg->ne[0], nt = sg->ne[1];
    const ggml_tensor * bias = ad->src[1];
    if (n_exp > 512 || nt < 1 || nt > 8 || sg->ne[2] != 1 || sg->type != GGML_TYPE_F32 || lg->type != GGML_TYPE_F32 ||
        bias->type != GGML_TYPE_F32 || ggml_nelements(bias) != n_exp || as->type != GGML_TYPE_I32 ||
        sg->nb[0] != sizeof(float) || lg->nb[0] != sizeof(float) || as->nb[0] != sizeof(int32_t)) return -1;
    auto root = [](const ggml_tensor * x) { while (x && x->view_src) x = x->view_src; return x; };
    const ggml_tensor * ids = gr->src[1];
    if (root(gr->src[0]) != sg || root(ids) != as || root(sr->src[0]) != gr || root(dv->src[0]) != gr || root(scl->src[0]) != dv) return -1;
    const int64_t k_used = ids->ne[0];
    if (k_used < 1 || k_used > 32 || k_used > n_exp || ids->ne[1] != nt || scl->type != GGML_TYPE_F32 ||
        scl->ne[0] != 1 || scl->ne[1] != k_used || scl->ne[2] != nt || ids->data != as->data) return -1;
    float sum_min, w_scale, w_bias = 0.0f;
    memcpy(&sum_min, cl->op_params, sizeof(float));
    memcpy(&w_scale, scl->op_params, sizeof(float));
    memcpy(&w_bias, (const int32_t *) scl->op_params + 1, sizeof(float));
    if (w_bias != 0.0f) return -1;
    { static bool said = false; if (!said) { said = true; fprintf(stderr, "[router] tail fused: %d experts, top %d, %d tokens\n", (int) n_exp, (int) k_used, (int) nt); } }
    moe_router_tail_k<<<(unsigned) nt, 512, 0, ctx.stream()>>>((const float *) lg->data, (const float *) bias->data, (float *) sg->data,
            (int32_t *) as->data, (float *) scl->data, (int) n_exp, (int) k_used, sum_min, w_scale,
            lg->nb[1] / sizeof(float), sg->nb[1] / sizeof(float), as->nb[1] / sizeof(int32_t), scl->nb[1] / sizeof(float), scl->nb[2] / sizeof(float));
    ggml_cuda_dre_node_done(ctx, as);
    return j;
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
        size_t m_nb0, size_t m_nb1, size_t d_nb0, size_t d_nb1, float * par_out) {
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
    if (par_out) {   // the caller continues the iterations in parallel (dsv4_hc_sinkhorn_par_k)
        for (int i = 0; i < n_hc*n_hc; ++i) par_out[t*256 + i] = c[i];
        return;
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

// One block per token, thread j owns row j in the row step and column j in the column step; the matrix is in shared
// memory. Each row / column sum adds its elements in the same order as the one-thread loop, so the result is the same.
static __global__ void dsv4_hc_sinkhorn_par_k(const float * c0, void * dst, int n_hc, int sinkhorn_iters, float eps,
                                               size_t d_nb0, size_t d_nb1) {
    __shared__ float c[16*16];
    const int64_t t = blockIdx.x;
    const int j = threadIdx.x;
    for (int i = j; i < n_hc*n_hc; i += blockDim.x) c[i] = c0[t*256 + i];
    __syncthreads();
    for (int it = 0; it < sinkhorn_iters - 1; ++it) {
        if (j < n_hc) {   // row j: sum over sh
            float s=0; for (int sh=0;sh<n_hc;++sh) s+=c[sh+j*n_hc];
            float inv=1.0f/(s+eps); for (int sh=0;sh<n_hc;++sh) c[sh+j*n_hc]*=inv;
        }
        __syncthreads();
        if (j < n_hc) {   // column j: sum over dh
            float s=0; for (int dh=0;dh<n_hc;++dh) s+=c[j+dh*n_hc];
            float inv=1.0f/(s+eps); for (int dh=0;dh<n_hc;++dh) c[j+dh*n_hc]*=inv;
        }
        __syncthreads();
    }
    for (int i = j; i < n_hc*n_hc; i += blockDim.x)
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
    // ALPHA_HC_SINKHORN_PAR (default 1): the iterations on n_hc threads per token; small token counts only (the scratch
    // is one fixed device buffer)
    static const bool par_on = [] { const char * e = getenv("ALPHA_HC_SINKHORN_PAR"); return !e || atoi(e) != 0; }();
    static float * par_buf = nullptr;
    constexpr int64_t PAR_MAX_TOK = 16384;
    const bool par = par_on && n_hc <= 16 && n_tokens <= PAR_MAX_TOK && si > 1;
    if (par && !par_buf) CUDA_CHECK(cudaMalloc(&par_buf, PAR_MAX_TOK * 256 * sizeof(float)));
    dsv4_hc_split_sinkhorn_k<<<(n_tokens+bs-1)/bs, bs, 0, ctx.stream()>>>(
        mixes->data, scale->data, base->data, dst->data,
        n_mix, n_tokens, n_hc, si, eps,
        mixes->nb[0], mixes->nb[1], dst->nb[0], dst->nb[1], par ? par_buf : nullptr);
    if (par) {
        dsv4_hc_sinkhorn_par_k<<<n_tokens, 32, 0, ctx.stream()>>>(par_buf, dst->data, n_hc, si, eps, dst->nb[0], dst->nb[1]);
    }

}
