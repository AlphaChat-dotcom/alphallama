#include "common.cuh"
#include "unary.cuh"
#include <mutex>
#include <utility>
#include <vector>
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

// Prompt-batch form (ALPHA_HC_PRE_BIG, default 1): one block of 512 threads per token.
static __global__ void dsv4_hc_pre_big_k(const float * __restrict__ x, const char * __restrict__ fn, const size_t fn_row,
        const float * __restrict__ scale, const float * __restrict__ base, float * __restrict__ split, float * __restrict__ out,
        const int n_embd, const int n_mix, const int sinkhorn_iters, const float eps, const float rms_eps,
        const size_t split_stride, const size_t out_stride) {
    constexpr int n_hc = 4;
    const int D = n_embd * n_hc;
    const int t = blockIdx.x;
    const float * xt = x + (size_t) t * D;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    __shared__ float sh_red[32][17];   // [logit row or 24 = sum of squares][warp]
    __shared__ float sh_mix[32];
    __shared__ float sh_c[16];
    __shared__ float sh_pre[4];
    const int n_blk = D / 32;
    float acc[24];
    float ss = 0.0f;
    for (int r = 0; r < n_mix; r++) acc[r] = 0.0f;
    for (int bk = tid; bk < n_blk; bk += blockDim.x) {
        float xv[32];
        for (int i = 0; i < 32; i++) { xv[i] = xt[bk * 32 + i]; ss += xv[i] * xv[i]; }
        for (int r = 0; r < n_mix; r++) {
            const char * blk = fn + (size_t) r * fn_row + (size_t) bk * 34;
            const float d = __half2float(*(const __half *) blk);
            const int8_t * qs = (const int8_t *) (blk + 2);
            float sd = 0.0f;
            for (int i = 0; i < 32; i++) sd += (float) qs[i] * xv[i];
            acc[r] += d * sd;
        }
    }
    for (int r = 0; r < n_mix; r++) {
        const float v = warp_reduce_sum(acc[r]);
        if (lane == 0) sh_red[r][warp] = v;
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) sh_red[24][warp] = ss;
    __syncthreads();
    if (tid < n_mix) {
        float tss = 0.0f, tot = 0.0f;
        for (int w = 0; w < nw; w++) { tss += sh_red[24][w]; tot += sh_red[tid][w]; }
        sh_mix[tid] = tot * rsqrtf(tss / D + rms_eps);
    }
    __syncthreads();
    if (tid < 16) {
        const float ps = scale[0], pos = scale[1], cs = scale[2];
        const int sh = tid & 3, dh = tid >> 2;
        float * dst = split + (size_t) t * split_stride;
        if (tid < n_hc) {
            const float pv = 1.0f / (1.0f + expf(-(sh_mix[tid] * ps + base[tid]))) + eps;
            sh_pre[tid] = pv;
            dst[tid] = pv;
            dst[n_hc + tid] = 2.0f / (1.0f + expf(-(sh_mix[n_hc + tid] * pos + base[n_hc + tid])));
        }
        float row[4];
        float mx = -INFINITY;
        for (int k = 0; k < n_hc; k++) {
            const int off = 2 * n_hc + k + dh * n_hc;
            row[k] = sh_mix[off] * cs + base[off];
            mx = fmaxf(mx, row[k]);
        }
        float sm = 0.0f;
        for (int k = 0; k < n_hc; k++) { row[k] = expf(row[k] - mx); sm += row[k]; }
        float c = row[sh] / sm + eps;
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
    __syncthreads();
    const float p0 = sh_pre[0], p1 = sh_pre[1], p2 = sh_pre[2], p3 = sh_pre[3];
    float * o = out + (size_t) t * out_stride;
    for (int d = tid; d < n_embd; d += blockDim.x) {
        float a = 0.0f;
        a += xt[d] * p0;
        a += xt[n_embd + d] * p1;
        a += xt[2 * n_embd + d] * p2;
        a += xt[3 * n_embd + d] * p3;
        o[d] = a;
    }
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
    if (n_hc != 4 || n_mix != (2 + n_hc) * n_hc || fnw->ne[0] != D || D % (32 * n_hc) != 0 || nt < 1 || nt > 65535) return -1;
    if (flat->type != GGML_TYPE_F32 || !ggml_is_contiguous(flat) || sp->type != GGML_TYPE_F32 || sp->nb[0] != sizeof(float)) return -1;
    if (flat->ne[2] != 1 || flat->ne[3] != 1 || sp->ne[0] != n_mix || sp->ne[1] != nt) return -1;
    if (!sp->src[1] || !sp->src[2] || sp->src[1]->type != GGML_TYPE_F32 || sp->src[2]->type != GGML_TYPE_F32) return -1;
    // views of the split, then the weighted sum over the same streams
    // several tokens: the pre view is made contiguous for the weighted sum (that CONT is covered too)
    const ggml_tensor * pre_c = nullptr;
    int j = i + 3;
    for (; j < cgraph->n_nodes && j < i + 12; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE) continue;
        if (n->op == GGML_OP_CONT && !pre_c && n->src[0] && n->src[0]->view_src == sp && n->src[0]->data == sp->data) { pre_c = n; continue; }
        break;
    }
    if (j >= cgraph->n_nodes) return -1;
    const ggml_tensor * ws = cgraph->nodes[j];
    if (ws->op != GGML_OP_DSV4_HC_WEIGHTED_SUM || !ws->src[0] || !ws->src[1]) return -1;
    const ggml_tensor * x = ws->src[0];
    const ggml_tensor * pre = pre_c ? pre_c->src[0] : ws->src[1];
    if (pre_c && ws->src[1] != pre_c) return -1;
    if (x->data != flat->data || !ggml_is_contiguous(x) || x->ne[1] != n_hc || x->ne[0] * n_hc != D || x->ne[2] != nt) return -1;
    if (pre->view_src != sp || pre->data != sp->data || ws->type != GGML_TYPE_F32 || ws->nb[0] != sizeof(float) || ws->ne[0] != x->ne[0]) return -1;
    static const bool big_on = [] { const char * e = getenv("ALPHA_HC_PRE_BIG"); return !e || atoi(e) != 0; }();
    if (nt > 64 || pre_c) {
        if (!big_on || D / 32 > 512 * 64) return -1;
        { static bool said = false; if (!said) { said = true; fprintf(stderr, "[hc-pre] prompt form: %d logits, %d tokens" "\n", (int) n_mix, (int) nt); } }
        dsv4_hc_pre_big_k<<<(unsigned) nt, 512, 0, ctx.stream()>>>((const float *) flat->data, (const char *) fnw->data, fnw->nb[1],
                (const float *) sp->src[1]->data, (const float *) sp->src[2]->data, (float *) sp->data, (float *) ws->data,
                (int) x->ne[0], (int) n_mix, si, eps, rms_eps, sp->nb[1] / sizeof(float), ws->nb[1] / sizeof(float));
        return j;
    }
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

// Indexer score (ALPHA_IDX_SCORE_FUSED, default 1).
//   kq [n_pool, n_tok, n_head] (contiguous matmul output), w [n_head, n_tok] -> out [n_pool, n_tok]
static __global__ void idx_score_k(const float * __restrict__ kq, const float * __restrict__ w, float * __restrict__ out,
        const int64_t n_pool, const int64_t n_tok, const int n_head, const size_t w_stride) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t plane = n_pool * n_tok;
    if (i >= plane) return;
    const int64_t t = i / n_pool;
    const float * wt = w + (size_t) t * w_stride;
    float s = 0.0f;
    for (int h = 0; h < n_head; h++) s += fmaxf(kq[i + (int64_t) h * plane], 0.0f) * wt[h];
    out[i] = s;
}

// nodes i..: CONT(permute(kq, 2,1,0,3)), UNARY relu, MUL by the head weights, SUM_ROWS, CONT(permute(., 2,1,0,3)).
// Returns the index of the last node covered, or -1.
int ggml_cuda_idx_score_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool on = [] { const char * e = getenv("ALPHA_IDX_SCORE_FUSED"); return !e || atoi(e) != 0; }();
    if (!on || i + 5 >= cgraph->n_nodes) return -1;
    const ggml_tensor * c1 = cgraph->nodes[i];
    if (c1->op != GGML_OP_CONT || !c1->src[0] || c1->src[0]->op != GGML_OP_PERMUTE) return -1;
    const ggml_tensor * kq = c1->src[0]->src[0];
    if (!kq || kq->op != GGML_OP_MUL_MAT || kq->type != GGML_TYPE_F32 || !ggml_is_contiguous(kq)) return -1;
    {   // ALPHA_IDX_SCORE_DBG: what follows a candidate, a few times
        static const bool dbg = getenv("ALPHA_IDX_SCORE_DBG") != nullptr;
        static int n_dbg = 0;
        if (dbg && n_dbg < 6 && kq->ne[2] >= 8) {
            n_dbg++;
            fprintf(stderr, "[idx-score] candidate kq [%lld,%lld,%lld]:", (long long) kq->ne[0], (long long) kq->ne[1], (long long) kq->ne[2]);
            for (int q = i; q < i + 8 && q < cgraph->n_nodes; q++) fprintf(stderr, " %s[%lld,%lld,%lld]", ggml_op_name(cgraph->nodes[q]->op),
                    (long long) cgraph->nodes[q]->ne[0], (long long) cgraph->nodes[q]->ne[1], (long long) cgraph->nodes[q]->ne[2]);
            fputc(10, stderr);
        }
    }
    const ggml_tensor * re = cgraph->nodes[i + 1];
    const ggml_tensor * mu = cgraph->nodes[i + 2];
    const ggml_tensor * sr = cgraph->nodes[i + 3];
    const ggml_tensor * pm = cgraph->nodes[i + 4];
    const ggml_tensor * c2 = cgraph->nodes[i + 5];
    if (re->op != GGML_OP_UNARY || (ggml_unary_op) ((const int32_t *) re->op_params)[0] != GGML_UNARY_OP_RELU || re->src[0] != c1) return -1;
    if (mu->op != GGML_OP_MUL || mu->src[0] != re || !mu->src[1]) return -1;
    if (sr->op != GGML_OP_SUM_ROWS || sr->src[0] != mu) return -1;
    if (pm->op != GGML_OP_PERMUTE || pm->src[0] != sr || c2->op != GGML_OP_CONT || c2->src[0] != pm) return -1;
    const ggml_tensor * w = mu->src[1];
    const int64_t n_pool = kq->ne[0], n_tok = kq->ne[1], n_head = kq->ne[2];
    if (kq->ne[3] != 1 || c1->ne[0] != n_head || c1->ne[1] != n_tok || c1->ne[2] != n_pool) return -1;
    if (w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != n_head || w->ne[1] != n_tok || w->ne[2] != 1) return -1;
    if (c2->type != GGML_TYPE_F32 || c2->ne[0] != n_pool || c2->ne[1] != n_tok || c2->ne[2] != 1 || !ggml_is_contiguous(c2)) return -1;
    if (n_head > 256) return -1;
    const int64_t plane = n_pool * n_tok;
    { static bool said = false; if (!said) { said = true; fprintf(stderr, "[idx-score] fused: %lld pools x %lld tokens x %lld heads\n", (long long) n_pool, (long long) n_tok, (long long) n_head); } }
    idx_score_k<<<(unsigned) ((plane + 255) / 256), 256, 0, ctx.stream()>>>((const float *) kq->data, (const float *) w->data, (float *) c2->data,
            n_pool, n_tok, (int) n_head, w->nb[1] / sizeof(float));
    return i + 5;
}

// ---- glm5-next linear-attention layer: conv and gate chains for prompt batches --------------------------------------
// ALPHA_GLM_CONV_FUSED (default 64 = minimum tokens, 0 = off)
static int glm_conv_min_tok() {
    static const int v = [] { const char * e = getenv("ALPHA_GLM_CONV_FUSED"); return e ? atoi(e) : 64; }();
    return v;
}
static const ggml_tensor * glm_root(const ggml_tensor * x) { while (x && x->view_src) x = x->view_src; return x; }
static bool glm_overlap(const void * a, size_t an, const void * b, size_t bn) {
    const char * a0 = (const char *) a, * b0 = (const char *) b;
    return a0 < b0 + bn && b0 < a0 + an;
}

struct glm_conv_plan {
    int i_ssm = -1, i_last = -1;
    const ggml_tensor * st = nullptr, * x = nullptr, * w = nullptr, * out = nullptr;
    bool norm = false;
    float eps = 0.0f, post = 1.0f;
    int64_t T = 0, C = 0, row = 0;
};

// cc = CONCAT(state view [3, C], transpose(x [C, T])) feeding SSM_CONV(w [4, C]) -> SILU [-> RMS_NORM -> SCALE].
// The same decision is taken at the CONCAT node (which then writes only its tail) and at the SSM_CONV node.
static bool glm_conv_plan_at(const ggml_cgraph * g, int i_cc, glm_conv_plan & P) {
    const int min_tok = glm_conv_min_tok();
    const ggml_tensor * cc = g->nodes[i_cc];
    if (min_tok <= 0 || cc->op != GGML_OP_CONCAT || ((const int32_t *) cc->op_params)[0] != 0 || cc->type != GGML_TYPE_F32) return false;
    const ggml_tensor * s = cc->src[0], * xt = cc->src[1];
    if (!s || !xt || s->type != GGML_TYPE_F32 || xt->type != GGML_TYPE_F32) return false;
    const int64_t T = xt->ne[0], C = xt->ne[1];
    if (T < min_tok || s->ne[0] != 3 || s->ne[1] != C || s->ne[2] != 1 || xt->ne[2] != 1 || cc->ne[0] != T + 3 || cc->ne[1] != C ||
        cc->ne[2] != 1 || cc->ne[3] != 1 || !ggml_is_contiguous(cc) || s->nb[0] != sizeof(float) || s->nb[1] != 3 * sizeof(float)) return false;
    const ggml_tensor * xr = glm_root(xt);
    if (!xr || xr == xt || xr->type != GGML_TYPE_F32 || !ggml_is_contiguous(xr) || xr->ne[0] != C || xr->ne[1] != T || ggml_nelements(xr) != C * T ||
        xt->data != xr->data || xt->nb[0] != (size_t) C * sizeof(float) || xt->nb[1] != sizeof(float)) return false;
    // consumers of the concat result: one SSM_CONV, and views of its 3 tail rows (the state update)
    int i_ssm = -1;
    for (int j = i_cc + 1; j < g->n_nodes; j++) {
        const ggml_tensor * n = g->nodes[j];
        if (n->view_src == cc) {
            if ((const char *) n->data != (const char *) cc->data + (size_t) T * sizeof(float) || n->ne[0] != 3 || n->ne[1] != C ||
                n->nb[0] != sizeof(float) || n->nb[1] != cc->nb[1] || ggml_nelements(n) != 3 * C) return false;
            continue;
        }
        if (n->src[0] == cc || n->src[1] == cc) {
            if (n->op != GGML_OP_SSM_CONV || n->src[0] != cc || i_ssm >= 0) return false;
            i_ssm = j;
        }
    }
    if (i_ssm < 0 || i_ssm + 1 >= g->n_nodes) return false;
    const ggml_tensor * ssm = g->nodes[i_ssm];
    const ggml_tensor * w = ssm->src[1];
    if (!w || w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != C || !ggml_is_contiguous(w) || ssm->ne[0] != C || ssm->ne[1] != T ||
        ssm->src[2] != xr) return false;   // the graph keeps x allocated up to here (the state is snapshotted at the CONCAT)
    const ggml_tensor * u = g->nodes[i_ssm + 1];
    if (u->op != GGML_OP_UNARY || (ggml_unary_op) ((const int32_t *) u->op_params)[0] != GGML_UNARY_OP_SILU || u->src[0] != ssm ||
        u->src[2] != xr || u->type != GGML_TYPE_F32 || !ggml_is_contiguous(u)) return false;
    P = glm_conv_plan();
    P.i_ssm = i_ssm; P.i_last = i_ssm + 1; P.st = s; P.x = xr; P.w = w; P.out = u; P.T = T; P.C = C;
    // optional l2 norm over rows of 128
    int k = i_ssm + 2;
    while (k < g->n_nodes && k < i_ssm + 6 && (g->nodes[k]->op == GGML_OP_RESHAPE || g->nodes[k]->op == GGML_OP_VIEW) && g->nodes[k]->view_src == u) k++;
    if (k + 1 < g->n_nodes) {
        const ggml_tensor * rn = g->nodes[k], * sc = g->nodes[k + 1];
        float sb[2] = { 0.0f, 0.0f };
        if (sc->op == GGML_OP_SCALE) memcpy(sb, sc->op_params, sizeof(sb));
        if (rn->op == GGML_OP_RMS_NORM && rn->src[0] && glm_root(rn->src[0]) == u && ggml_is_contiguous(rn->src[0]) && rn->ne[0] == 128 && C % 128 == 0 &&
            rn->src[2] == xr && sc->op == GGML_OP_SCALE && sc->src[0] == rn && sc->src[2] == xr && sb[1] == 0.0f &&
            sc->type == GGML_TYPE_F32 && ggml_is_contiguous(sc) && ggml_nelements(sc) == C * T) {
            bool sole = true;   // nothing else may read the un-normed activation or the RMS output
            for (int j = i_ssm + 2; j < g->n_nodes && sole; j++) {
                const ggml_tensor * n = g->nodes[j];
                if (n == rn || n == sc || n->view_src == u) continue;
                for (int q = 0; q < 2; q++) {
                    if (n->src[q] && (glm_root(n->src[q]) == u || n->src[q] == rn)) sole = false;
                }
            }
            if (sole) {
                P.norm = true; P.out = sc; P.i_last = k + 1; P.row = 128; P.post = sb[0];
                memcpy(&P.eps, rn->op_params, sizeof(float));
            }
        }
    }
    // x must stay untouched from the CONCAT to the kernel (nothing in between may write over it), and the output must not alias it
    const size_t xn = ggml_nbytes(xr);
    for (int j = i_cc; j <= P.i_last; j++) {
        const ggml_tensor * n = g->nodes[j];
        if (!n->data || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE) continue;
        if (glm_overlap(n->data, ggml_nbytes(n), xr->data, xn)) return false;
    }
    return true;
}

static __device__ __forceinline__ float glm_conv_tap(const float * __restrict__ x, const float * __restrict__ st,
        const float * __restrict__ w, const int64_t C, const int c, const int64_t t) {
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int64_t m = t + j;
        const float v = m < 3 ? st[m + (int64_t) c * 3] : x[c + (m - 3) * C];
        sum += v * w[j + c * 4];
    }
    return ggml_cuda_op_silu_single(sum);
}

// out[c, t] = silu(sum_j w[j, c] * in[t + j, c]),  in = the 3 state rows followed by x
static __global__ void glm_conv_silu_k(const float * __restrict__ x, const float * __restrict__ st, const float * __restrict__ w,
        float * __restrict__ out, const int64_t C, const int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C * T) return;
    out[i] = glm_conv_tap(x, st, w, C, (int) (i % C), i / C);
}

// the same followed by out = v * post / sqrt(mean_row(v^2) + eps) over rows of 128 channels; one warp per row
static __global__ void glm_conv_silu_norm_k(const float * __restrict__ x, const float * __restrict__ st, const float * __restrict__ w,
        float * __restrict__ out, const int64_t C, const int64_t T, const float eps, const float post) {
    const int lane = threadIdx.x & 31;
    const int64_t r = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int64_t n_row = C / 128;
    if (r >= n_row * T) return;
    const int64_t t = r / n_row;
    const int c0 = (int) (r % n_row) * 128 + lane;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int e = 0; e < 4; e++) { v[e] = glm_conv_tap(x, st, w, C, c0 + 32 * e, t); ss += v[e] * v[e]; }
    ss = warp_reduce_sum(ss);
    const float sc = post / sqrtf(ss / 128.0f + eps);
#pragma unroll
    for (int e = 0; e < 4; e++) out[c0 + 32 * e + t * C] = v[e] * sc;
}

// the concat result's 3 tail rows (what the state update reads): cc[T + j, c] = x[c, T - 3 + j]; and a snapshot of the
// entering state for the conv kernel (the state update that follows may write the same cells)
static __global__ void glm_conv_tail_k(const float * __restrict__ x, const float * __restrict__ st, float * __restrict__ cc,
        float * __restrict__ snap, const int64_t C, const int64_t T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * C) return;
    const int c = i / 3, j = i % 3;
    snap[i] = st[i];
    cc[T + j + (int64_t) c * (T + 3)] = x[c + (T - 3 + j) * C];
}

// state snapshots between a CONCAT and its SSM_CONV (q, k and v of a layer are open at the same time)
struct glm_conv_snaps {
    std::mutex mu;
    std::vector<std::pair<const ggml_tensor *, float *>> open;
    std::vector<std::pair<float *, size_t>> spare;
};
static glm_conv_snaps g_glm_snaps;

bool ggml_cuda_glm_conv_concat(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    glm_conv_plan P;
    if (!glm_conv_plan_at(cgraph, i, P)) return false;
    const ggml_tensor * cc = cgraph->nodes[i];
    const size_t need = (size_t) 3 * P.C * sizeof(float);
    float * snap = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_glm_snaps.mu);
        for (size_t q = 0; q < g_glm_snaps.open.size(); q++) {   // a stale entry of an abandoned graph
            if (g_glm_snaps.open[q].first == cc) { snap = g_glm_snaps.open[q].second; g_glm_snaps.open.erase(g_glm_snaps.open.begin() + q); break; }
        }
        if (snap) { g_glm_snaps.spare.push_back(std::make_pair(snap, need)); snap = nullptr; }
        for (size_t q = 0; q < g_glm_snaps.spare.size(); q++) {
            if (g_glm_snaps.spare[q].second >= need) { snap = g_glm_snaps.spare[q].first; g_glm_snaps.spare.erase(g_glm_snaps.spare.begin() + q); break; }
        }
        if (!snap) CUDA_CHECK(cudaMalloc(&snap, need));
        g_glm_snaps.open.push_back(std::make_pair(cc, snap));
    }
    glm_conv_tail_k<<<(unsigned) ((3 * P.C + 255) / 256), 256, 0, ctx.stream()>>>((const float *) P.x->data, (const float *) P.st->data,
            (float *) cc->data, snap, P.C, P.T);
    return true;
}

// Returns the index of the last node covered, or -1.
int ggml_cuda_glm_conv_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * ssm = cgraph->nodes[i];
    const ggml_tensor * cc = ssm->src[0];
    if (glm_conv_min_tok() <= 0 || !cc || cc->op != GGML_OP_CONCAT || ssm->ne[1] < glm_conv_min_tok()) return -1;
    int i_cc = -1;
    for (int j = i - 1; j >= 0; j--) if (cgraph->nodes[j] == cc) { i_cc = j; break; }
    glm_conv_plan P;
    if (i_cc < 0 || !glm_conv_plan_at(cgraph, i_cc, P) || P.i_ssm != i) return -1;
    float * snap = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_glm_snaps.mu);
        for (size_t q = 0; q < g_glm_snaps.open.size(); q++) {
            if (g_glm_snaps.open[q].first == cc) {
                snap = g_glm_snaps.open[q].second;
                g_glm_snaps.open.erase(g_glm_snaps.open.begin() + q);
                g_glm_snaps.spare.push_back(std::make_pair(snap, (size_t) 3 * P.C * sizeof(float)));   // free for reuse once this kernel is queued
                break;
            }
        }
    }
    GGML_ASSERT(snap != nullptr);   // the CONCAT of this conv took the same decision
    { static bool said = false; if (!said) { said = true; fprintf(stderr, "[glm-conv] fused: %lld channels x %lld tokens, l2 norm %d" "\n", (long long) P.C, (long long) P.T, (int) P.norm); } }
    if (P.norm) {
        glm_conv_silu_norm_k<<<(unsigned) ((P.C / 128 * P.T + 3) / 4), 128, 0, ctx.stream()>>>((const float *) P.x->data, snap,
                (const float *) P.w->data, (float *) P.out->data, P.C, P.T, P.eps, P.post);
    } else {
        glm_conv_silu_k<<<(unsigned) ((P.C * P.T + 255) / 256), 256, 0, ctx.stream()>>>((const float *) P.x->data, snap,
                (const float *) P.w->data, (float *) P.out->data, P.C, P.T);
    }
    return P.i_last;
}

// Decay gate: out = s2 * sigmoid(s1 * ((g + dt[c]) * a[c / head_dim]))
static __global__ void glm_gate_chain_k(const float * __restrict__ g, const float * __restrict__ dt, const float * __restrict__ a,
        float * __restrict__ out, const int64_t C, const int64_t n, const int head_dim, const float s1, const float s2) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = (int) (i % C);
    const float v = ((g[i] + dt[c]) * a[c / head_dim]) * s1;
    out[i] = (1.0f / (1.0f + expf(-v))) * s2;
}

// nodes i..: ADD(g, dt), MUL by the per-head factor, SCALE, SIGMOID, SCALE (reshapes between). Returns the last index or -1.
int ggml_cuda_glm_gate_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const int min_tok = glm_conv_min_tok();
    const ggml_tensor * ad = cgraph->nodes[i];
    if (min_tok <= 0 || ad->op != GGML_OP_ADD || ad->type != GGML_TYPE_F32 || !ad->src[0] || !ad->src[1]) return -1;
    const ggml_tensor * gt = ad->src[0], * dt = ad->src[1];
    const int64_t C = ad->ne[0], T = ad->ne[1];
    if (T < min_tok || ad->ne[2] != 1 || ad->ne[3] != 1 || !ggml_is_contiguous(ad) || !ggml_are_same_shape(gt, ad) || gt->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(gt) || dt->type != GGML_TYPE_F32 || ggml_nelements(dt) != C || !ggml_is_contiguous(dt)) return -1;
    const ggml_tensor * mu = nullptr, * s1 = nullptr, * un = nullptr, * s2 = nullptr;
    int j = i + 1;
    for (; j < cgraph->n_nodes && j < i + 12 && !s2; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        switch (n->op) {
            case GGML_OP_RESHAPE: case GGML_OP_VIEW: break;
            case GGML_OP_MUL:   if (mu || !n->src[0] || glm_root(n->src[0]) != ad || !n->src[1]) return -1; mu = n; break;
            case GGML_OP_SCALE: if (!mu) return -1;
                                if (!s1) { if (n->src[0] != mu) return -1; s1 = n; }
                                else     { if (!un || n->src[0] != un) return -1; s2 = n; }
                                break;
            case GGML_OP_UNARY: if (!s1 || un || n->src[0] != s1 || (ggml_unary_op) ((const int32_t *) n->op_params)[0] != GGML_UNARY_OP_SIGMOID) return -1; un = n; break;
            default: return -1;
        }
    }
    if (!s2) return -1;
    j--;
    const ggml_tensor * a = mu->src[1];
    float p1[2], p2[2];
    memcpy(p1, s1->op_params, sizeof(p1));
    memcpy(p2, s2->op_params, sizeof(p2));
    const int64_t n_head = mu->ne[1], head_dim = mu->ne[0];
    if (a->type != GGML_TYPE_F32 || ggml_nelements(a) != n_head || a->ne[1] != n_head || head_dim * n_head != C || ggml_nelements(mu) != C * T ||
        p1[1] != 0.0f || p2[1] != 0.0f || s2->type != GGML_TYPE_F32 || !ggml_is_contiguous(s2) || ggml_nelements(s2) != C * T) return -1;
    for (int q = j + 1; q < cgraph->n_nodes; q++) {   // the intermediates have no other readers
        const ggml_tensor * n = cgraph->nodes[q];
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            const ggml_tensor * r = n->src[k] ? glm_root(n->src[k]) : nullptr;
            if (r && (r == ad || r == mu || r == s1 || r == un)) return -1;
        }
    }
    const size_t nb = (size_t) C * T * sizeof(float);
    if (s2->data != gt->data && glm_overlap(s2->data, nb, gt->data, nb)) return -1;
    if (glm_overlap(s2->data, nb, dt->data, ggml_nbytes(dt)) || glm_overlap(s2->data, nb, a->data, ggml_nbytes(a))) return -1;
    { static bool said = false; if (!said) { said = true; fprintf(stderr, "[glm-gate] fused: %lld x %lld, scales %g %g" "\n", (long long) C, (long long) T, p1[0], p2[0]); } }
    glm_gate_chain_k<<<(unsigned) ((C * T + 255) / 256), 256, 0, ctx.stream()>>>((const float *) gt->data, (const float *) dt->data,
            (const float *) a->data, (float *) s2->data, C, C * T, (int) head_dim, p1[0], p2[0]);
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
