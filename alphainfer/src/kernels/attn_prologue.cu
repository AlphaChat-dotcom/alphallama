// Attention prologue in one launch (AI_ATTN_PRO): q/k head RMS norms, multi-section RoPE, the K and V cache
// writes and the gate's contiguous copy.
//
// Replaces, per attention layer at decode: rms_norm_f32<256> x2 (q, k), rope_multi x2, k_set_rows x2,
// and the gate cont -- seven serial latency-bound kernels between the qkv projection and flash attention.
//
// Bit-exact by construction:
//   * norm: rms_norm_f32<256, do_multiply> -- one block per (token, head) row, `tmp += x*x` (contracted), the
//     ggml's block_reduce (exact.cuh block_sum<256>), mean by the approximate division, rsqrtf, scale*x*w;
//   * rope: rope_multi's arithmetic with ext_factor 0 and attn_factor 1, copied: theta_base = pos * powf(theta_scale,
//     i0/2), theta = freq_scale * (theta_base / 1), cos/sin, pairs (j, j + n_dims/2), dims >= n_dims copied;
//   * cache writes: the float -> half conversion of k_set_rows; gate: a plain copy.
// This file is built -use_fast_math, as ggml-cuda is, so powf/cosf/sinf lower the same way.
// AI_ATTN_PRO=2 is the instrument: the ggml chain runs and is consumed, this kernel writes shadows, and q,
// gate, K rows and V rows are compared bitwise.
#include "kernels/kernels.h"
#include "kernels/exact.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int AP_D = 256;

struct ap_sections {
    int v[4];
};

template <bool forward>
__device__ __forceinline__ void ap_rope_yarn(const float theta_extrap, const float freq_scale, const float ext_factor,
                                             float mscale, float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta        = theta_interp;
    (void) ext_factor;   // 0 at every call site this kernel accepts (the host checks)
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

__device__ __forceinline__ float ap_theta_base(const int32_t * pos, int t, int T, int i0, float theta_scale,
                                               const ap_sections & sections, bool is_imrope) {
    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;
    float     theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {
            theta_base = pos[t + T * 1] * powf(theta_scale, i0 / 2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {
            theta_base = pos[t + T * 2] * powf(theta_scale, i0 / 2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {
            theta_base = pos[t] * powf(theta_scale, i0 / 2.0f);
        } else {
            theta_base = pos[t + T * 3] * powf(theta_scale, i0 / 2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[t] * powf(theta_scale, i0 / 2.0f);
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[t + T * 1] * powf(theta_scale, i0 / 2.0f);
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[t + T * 2] * powf(theta_scale, i0 / 2.0f);
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[t + T * 3] * powf(theta_scale, i0 / 2.0f);
        }
    }
    return theta_base;
}

// grid (T, Hq + 2*Hkv), block AP_D
__global__ void __launch_bounds__(AP_D) attn_prologue_k(
        const float * __restrict__ qg, long long qg_row, const float * __restrict__ kp, long long k_row,
        const float * __restrict__ vp, long long v_row, int Hq, int Hkv, int T, const float * __restrict__ qw,
        const float * __restrict__ kw, float eps, const int32_t * __restrict__ pos, int n_dims, ap_sections sections,
        int is_imrope, float theta_scale, float freq_scale, const int64_t * __restrict__ cells, half * __restrict__ kc,
        long long kc_row, half * __restrict__ vc, long long vc_row, float * __restrict__ q_out,
        float * __restrict__ gate_out, int rows_by_token) {
    const int t = blockIdx.x, y = blockIdx.y, tid = threadIdx.x;
    const long long cr = rows_by_token ? (long long) t : (long long) cells[t];   // shadow rows for the compare
    // q|gate interleaved per head, k and v: one fused projection row or three separate projections
    const float * qrow = qg + (long long) t * qg_row;
    const float * krow = kp + (long long) t * k_row;
    const float * vrow = vp + (long long) t * v_row;
    if (y >= Hq + Hkv) {   // V: f32 -> f16 into the cache row
        const int h = y - Hq - Hkv;
        vc[cr * vc_row + (long long) h * AP_D + tid] = __float2half(vrow[(long long) h * AP_D + tid]);
        return;
    }
    const bool  is_q = y < Hq;
    const int   h    = is_q ? y : y - Hq;
    const float x    = is_q ? qrow[(long long) h * 2 * AP_D + tid] : krow[(long long) h * AP_D + tid];
    if (is_q) {
        gate_out[((long long) t * Hq + h) * AP_D + tid] = qrow[(long long) h * 2 * AP_D + AP_D + tid];
    }
    // rms_norm_f32<256, do_multiply>
    __shared__ float s_sum[32];
    __shared__ float ys[AP_D];
    float            tmp = 0.0f;
    tmp += x * x;
    tmp                  = ai::ex::block_sum<AP_D>(tmp, s_sum, tid);
    const float scale    = ai::ex::rms_scale(tmp, AP_D, eps);
    ys[tid]              = ai::ex::rms_apply(scale, x, is_q ? qw[tid] : kw[tid]);
    __syncthreads();
    // rope_multi
    float o0 = 0.0f, o1 = 0.0f;
    int   p0 = -1, p1 = -1;
    if (tid < n_dims / 2) {
        const int   i0 = 2 * tid;
        const float tb = ap_theta_base(pos, t, T, i0, theta_scale, sections, is_imrope != 0);
        float       cos_theta, sin_theta;
        ap_rope_yarn<true>(tb / 1.0f, freq_scale, 0.0f, 1.0f, cos_theta, sin_theta);
        const float x0 = ys[tid];
        const float x1 = ys[tid + n_dims / 2];
        o0             = x0 * cos_theta - x1 * sin_theta;
        o1             = x0 * sin_theta + x1 * cos_theta;
        p0             = tid;
        p1             = tid + n_dims / 2;
    }
    if (is_q) {
        float * q = q_out + ((long long) t * Hq + h) * AP_D;
        if (p0 >= 0) {
            q[p0] = o0;
            q[p1] = o1;
        }
        if (tid >= n_dims) {
            q[tid] = ys[tid];
        }
    } else {
        half * k = kc + cr * kc_row + (long long) h * AP_D;
        if (p0 >= 0) {
            k[p0] = __float2half(o0);
            k[p1] = __float2half(o1);
        }
        if (tid >= n_dims) {
            k[tid] = __float2half(ys[tid]);
        }
    }
}

// AI_ATTN_OGATE: the attention output gate and the o-projection's q8_1 copy in one pass. ggml fuses
// mul(attn, sigmoid(gate)) into unary_gated_op_kernel<op_sigmoid>: dst = op_sigmoid(gate) * attn with
// op_sigmoid = 1/(1+expf(-x)) (this file is -use_fast_math, as unary.cu). The q8_1 copy is exact.cuh's
// emitter (byte-identical to ggml's quantize_q8_1), registered so the o-projection skips its quantise.
__global__ void attn_ogate_q8_k(const float * __restrict__ attn, const float * __restrict__ gate, float * __restrict__ out,
                                ai::ex::block_q8_1 * __restrict__ q8, long long ncols) {
    const long long t   = blockIdx.y;
    const long long col = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long i   = t * ncols + col;
    const float     g   = gate[i];
    const float     o   = (1.0f / (1.0f + expf(-g))) * attn[i];
    out[i]              = o;
    if (q8 != nullptr) {
        ai::ex::emit_q8_1(q8, i, o);
    }
}

__device__ unsigned long long g_ap_cmp[3];
__global__ void ap_cmp_f32_k(const float * a, const float * b, long long n) {
    unsigned long long d = 0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        d += __float_as_uint(a[i]) != __float_as_uint(b[i]);
    }
    atomicAdd(&g_ap_cmp[2], d);
    if (threadIdx.x == 0) {
        atomicAdd(&g_ap_cmp[0], 1ull);
        atomicAdd(&g_ap_cmp[1], (unsigned long long) n);
    }
}
// shadow rows [T][width] f16 vs cache rows at cells[t]
__global__ void ap_cmp_rows_k(const half * a, const half * cache, long long row, const int64_t * cells, int T, int width) {
    unsigned long long d = 0;
    for (long long i = threadIdx.x; i < (long long) T * width; i += blockDim.x) {
        const int t = (int) (i / width), c = (int) (i % width);
        d += __half_as_ushort(a[i]) != __half_as_ushort(cache[cells[t] * row + c]);
    }
    atomicAdd(&g_ap_cmp[2], d);
    if (threadIdx.x == 0) {
        atomicAdd(&g_ap_cmp[0], 1ull);
        atomicAdd(&g_ap_cmp[1], (unsigned long long) T * width);
    }
}

}   // namespace

namespace ai::kern {

bool attn_prologue_ok(int64_t D, int64_t n_dims) {
    return D == AP_D && n_dims > 0 && n_dims <= AP_D && n_dims % 2 == 0;
}

bool attn_prologue3(void * stream, const float * qg, int64_t qg_row, const float * kp, int64_t k_row, const float * vp,
                    int64_t v_row, int64_t Hq, int64_t Hkv, int64_t T, const float * qw, const float * kw, float eps,
                    const int32_t * pos, int n_dims, const int * sections, bool is_imrope, float freq_base,
                    float freq_scale, const int64_t * cells, void * kc, int64_t kc_row, void * vc, int64_t vc_row,
                    float * q_out, float * gate_out, bool rows_by_token) {
    if (!attn_prologue_ok(AP_D, n_dims) || T < 1) {
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[attn-pro] LAUNCHED: q/k norm + rope + K/V cache writes + gate copy in one kernel\n");
    }
    ap_sections sec;
    for (int i = 0; i < 4; ++i) {
        sec.v[i] = sections[i];
    }
    const float theta_scale = powf(freq_base, -2.0f / n_dims);   // as rope_multi_cuda, on the host
    const dim3  grid((unsigned) T, (unsigned) (Hq + 2 * Hkv));
    attn_prologue_k<<<grid, AP_D, 0, (cudaStream_t) stream>>>(qg, (long long) qg_row, kp, (long long) k_row, vp,
                                                              (long long) v_row, (int) Hq, (int) Hkv, (int) T, qw, kw,
                                                              eps, pos, n_dims, sec, is_imrope ? 1 : 0, theta_scale,
                                                              freq_scale, cells, (half *) kc, (long long) kc_row,
                                                              (half *) vc, (long long) vc_row, q_out, gate_out,
                                                              rows_by_token ? 1 : 0);
    return true;
}

bool attn_prologue(void * stream, const float * fo, int64_t fo_row, int64_t Hq, int64_t Hkv, int64_t T, const float * qw,
                   const float * kw, float eps, const int32_t * pos, int n_dims, const int * sections, bool is_imrope,
                   float freq_base, float freq_scale, const int64_t * cells, void * kc, int64_t kc_row, void * vc,
                   int64_t vc_row, float * q_out, float * gate_out, bool rows_by_token) {
    const int64_t qd = 2 * AP_D * Hq, kd = AP_D * Hkv;
    return attn_prologue3(stream, fo, fo_row, fo + qd, fo_row, fo + qd + kd, fo_row, Hq, Hkv, T, qw, kw, eps, pos, n_dims,
                          sections, is_imrope, freq_base, freq_scale, cells, kc, kc_row, vc, vc_row, q_out, gate_out,
                          rows_by_token);
}

bool attn_ogate_q8(void * stream, const float * attn, const float * gate, float * out, void * q8, int64_t ncols, int64_t T) {
    if (ncols % 256 != 0 || (q8 != nullptr && !ai::ex::q8_geometry_ok(ncols, 256))) {
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[attn-ogate] LAUNCHED: sigmoid gate + o-proj q8_1 copy in one kernel (q8 %s)\n", q8 ? "STAGED" : "off");
    }
    attn_ogate_q8_k<<<dim3((unsigned) (ncols / 256), (unsigned) T), 256, 0, (cudaStream_t) stream>>>(
        attn, gate, out, (ai::ex::block_q8_1 *) q8, (long long) ncols);
    return true;
}

void attn_ogate_compare(void * stream, const float * a, const float * b, int64_t n) {
    static bool armed = false;
    if (!armed) {
        armed = true;
        atexit([]() {
            unsigned long long h[3] = { 0, 0, 0 };
            cudaMemcpyFromSymbol(h, g_ap_cmp, sizeof(h));
            fprintf(stderr, "[attn-ogate-compare] arrays %llu | values %llu, differing %llu\n", h[0], h[1], h[2]);
        });
    }
    ap_cmp_f32_k<<<1, 256, 0, (cudaStream_t) stream>>>(a, b, (long long) n);
}

void attn_prologue_compare(void * stream, const float * q_a, const float * q_b, int64_t nq, const float * g_a,
                           const float * g_b, int64_t ng, const void * k_a, const void * kc, int64_t kc_row,
                           const void * v_a, const void * vc, int64_t vc_row, const int64_t * cells, int64_t T,
                           int64_t width) {
    static bool armed = false;
    if (!armed) {
        armed = true;
        atexit([]() {
            unsigned long long h[3] = { 0, 0, 0 };
            cudaMemcpyFromSymbol(h, g_ap_cmp, sizeof(h));
            fprintf(stderr, "[attn-pro-compare] arrays %llu | values %llu, differing %llu\n", h[0], h[1], h[2]);
        });
    }
    cudaStream_t st = (cudaStream_t) stream;
    ap_cmp_f32_k<<<1, 256, 0, st>>>(q_a, q_b, (long long) nq);
    ap_cmp_f32_k<<<1, 256, 0, st>>>(g_a, g_b, (long long) ng);
    ap_cmp_rows_k<<<1, 256, 0, st>>>((const half *) k_a, (const half *) kc, (long long) kc_row, cells, (int) T, (int) width);
    ap_cmp_rows_k<<<1, 256, 0, st>>>((const half *) v_a, (const half *) vc, (long long) vc_row, cells, (int) T, (int) width);
}

}   // namespace ai::kern
