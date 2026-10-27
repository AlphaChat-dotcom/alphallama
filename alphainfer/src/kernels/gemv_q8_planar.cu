// Q8_0 decode GEMV with no activation quantisation: f32 activations, weights dequantised in
// registers, fp32 FMA.
//
// The MMVQ path quantises every matmul input to q8_1 first, which costs a quantize launch per matmul
// input, and quantising activations differently (static, delayed, rotated scales) costs quality.
// These kernels skip the quantise entirely; they pay off only if they match MMVQ's bandwidth.
//
// Layout: planar, repacked from GGUF's interleaved Q8_0 at the same byte count:
//   codes  int8  [rows][cols]      contiguous, so each lane takes one aligned 8-byte load
//   scales fp16  [rows][cols/32]   immediately after the codes
// GGUF's block is 2 B scale + 32 B codes = 34 B, which is never 16-byte aligned, so vector loads of
// the codes are impossible in place. 32 + 2 bytes per 32 weights either way, so the repack is in place.
//
// Schedule: one warp per R rows. Per step a warp covers 256 codes (8 Q8_0 blocks): lane L owns the
// 8 consecutive codes [256*s + 8L, +8), so the warp's code read is one coalesced 256-byte span and a
// lane's 8 codes never straddle a 32-code block (one scale per lane per step). x is staged once per
// CTA in shared memory and each lane's 8 x values are held in registers across the R rows, so the
// on-chip x traffic is amortised R-fold. Accumulation is fp32; the per-block scale is applied to the
// block's partial dot product, the same association GGUF's own dequantise-then-dot uses.
#include "kernels/kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>

namespace {

constexpr int kWarp       = 32;
constexpr int kWarpsPerCta = 4;
constexpr int kStep       = 256;   // codes per warp step: 32 lanes x 8

template <int R>
__global__ void gemv_q8_planar_k(const int8_t * __restrict__ codes, const __half * __restrict__ scales,
                                 const float * __restrict__ x, float * __restrict__ y, int rows, int cols) {
    extern __shared__ float xs[];
    const int tid = threadIdx.y * kWarp + threadIdx.x;
    for (int i = tid * 4; i < cols; i += kWarpsPerCta * kWarp * 4) {
        *reinterpret_cast<float4 *>(xs + i) = *reinterpret_cast<const float4 *>(x + i);
    }
    __syncthreads();

    const int lane = threadIdx.x;
    const int row0 = (blockIdx.x * kWarpsPerCta + threadIdx.y) * R;
    const int nb   = cols >> 5;

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] = 0.0f;
    }
    for (int base = lane * 8; base < cols; base += kStep) {
        const float4 xa  = *reinterpret_cast<const float4 *>(xs + base);
        const float4 xb  = *reinterpret_cast<const float4 *>(xs + base + 4);
        const int    blk = base >> 5;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row < rows) {
                const int2 v = *reinterpret_cast<const int2 *>(codes + (size_t) row * cols + base);
                const float d = __half2float(scales[(size_t) row * nb + blk]);
                const int8_t * c = reinterpret_cast<const int8_t *>(&v);
                float s = (float) c[0] * xa.x;
                s = fmaf((float) c[1], xa.y, s);
                s = fmaf((float) c[2], xa.z, s);
                s = fmaf((float) c[3], xa.w, s);
                s = fmaf((float) c[4], xb.x, s);
                s = fmaf((float) c[5], xb.y, s);
                s = fmaf((float) c[6], xb.z, s);
                s = fmaf((float) c[7], xb.w, s);
                acc[r] = fmaf(d, s, acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o, kWarp);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r < rows) {
                y[row0 + r] = acc[r];
            }
        }
    }
}

// Split-K variant for wide tensors with few rows, where the R-per-warp kernel underfills the GPU
// (R=4 leaves too few CTAs, and R=1 pays an x-staging barrier per 4 rows). Here S warps share one
// row, each taking a contiguous 1/S of K, and a CTA of 4 warps covers 4/S rows, so rows=2048 at S=4
// launches 2048 CTAs.
// No shared-memory staging of x: a lane's 8 consecutive x values are one 32-byte span and the
// warp's 32 lanes read 1 KB contiguous, which is already fully coalesced from L1/L2. The only
// shared memory is S partial sums per row, reduced after one barrier.
//
// T columns (T <= 8): a lane loads its 8 codes and their scale once and dequantises them once, then
// reuses them against all T activation columns -- so weight bytes and dequant work are paid once
// whatever T is, and only the (small, L1/L2-resident) x reads scale with T. x and y are column-major
// contiguous: column t of x at x + t*cols, of y at y + t*rows.
//
// The activation element type XT is float or __half. With f32 x the kernel is not flat in T on Ampere:
// x costs 4 bytes per weight byte per column, re-read for every row, so at T=8 it asks ~32x the DRAM
// bandwidth of L1, more than the hardware ratio. fp16 x halves that. Accumulation stays fp32 either
// way; only the load differs.
__device__ __forceinline__ void load8(const float * p, float (&v)[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4 *>(p));
    const float4 b = __ldg(reinterpret_cast<const float4 *>(p + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void load8(const __half * p, float (&v)[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4 *>(p));   // 8 halves, one 16-byte load
    const __half2 * h = reinterpret_cast<const __half2 *>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        v[2 * i]     = f.x;
        v[2 * i + 1] = f.y;
    }
}

template <int S, int T, typename XT>
__global__ void gemv_q8_planar_splitk_k(const int8_t * __restrict__ codes, const __half * __restrict__ scales,
                                        const XT * __restrict__ x, float * __restrict__ y, int rows, int cols) {
    __shared__ float part[kWarpsPerCta][T];
    const int lane  = threadIdx.x;
    const int w     = threadIdx.y;
    const int row   = blockIdx.x * (kWarpsPerCta / S) + w / S;
    const int piece = w % S;
    const int span  = cols / S;
    const int k0    = piece * span;
    const int nb    = cols >> 5;

    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        acc[t] = 0.0f;
    }
    if (row < rows) {
        const int8_t * crow = codes + (size_t) row * cols;
        const __half * srow = scales + (size_t) row * nb;
        for (int base = k0 + lane * 8; base < k0 + span; base += kStep) {
            const int2   v = *reinterpret_cast<const int2 *>(crow + base);
            const float  d = __half2float(srow[base >> 5]);
            const int8_t * c = reinterpret_cast<const int8_t *>(&v);
            float cw[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                cw[j] = (float) c[j];   // dequantised once, reused for every column below
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                float xv[8];
                load8(x + (size_t) t * cols + base, xv);
                float s = cw[0] * xv[0];
#pragma unroll
                for (int j = 1; j < 8; ++j) {
                    s = fmaf(cw[j], xv[j], s);
                }
                acc[t] = fmaf(d, s, acc[t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o, kWarp);
        }
    }
    if (S == 1) {
        if (lane == 0 && row < rows) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                y[(size_t) t * rows + row] = acc[t];
            }
        }
        return;
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            part[w][t] = acc[t];
        }
    }
    __syncthreads();
    if (lane == 0 && piece == 0 && row < rows) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float sum = 0.0f;
#pragma unroll
            for (int p = 0; p < S; ++p) {
                sum += part[w + p][t];
            }
            y[(size_t) t * rows + row] = sum;
        }
    }
}

template <int S, int T, typename XT>
void launch_splitk(cudaStream_t st, const uint8_t * w, int64_t rows, int64_t cols, const XT * x, float * y) {
    const int8_t * codes  = reinterpret_cast<const int8_t *>(w);
    const __half * scales = reinterpret_cast<const __half *>(w + rows * cols);
    const int64_t  rows_per_cta = kWarpsPerCta / S;
    const dim3     grid((unsigned) ((rows + rows_per_cta - 1) / rows_per_cta));
    const dim3     block(kWarp, kWarpsPerCta);
    gemv_q8_planar_splitk_k<S, T, XT><<<grid, block, 0, st>>>(codes, scales, x, y, (int) rows, (int) cols);
}

template <int S, typename XT>
void launch_splitk_t(cudaStream_t st, const uint8_t * w, int64_t rows, int64_t cols, int64_t ncols,
                     const XT * x, float * y) {
    switch (ncols) {
        case 1: launch_splitk<S, 1, XT>(st, w, rows, cols, x, y); break;
        case 2: launch_splitk<S, 2, XT>(st, w, rows, cols, x, y); break;
        case 3: launch_splitk<S, 3, XT>(st, w, rows, cols, x, y); break;
        case 4: launch_splitk<S, 4, XT>(st, w, rows, cols, x, y); break;
        case 5: launch_splitk<S, 5, XT>(st, w, rows, cols, x, y); break;
        case 6: launch_splitk<S, 6, XT>(st, w, rows, cols, x, y); break;
        case 7: launch_splitk<S, 7, XT>(st, w, rows, cols, x, y); break;
        case 8: launch_splitk<S, 8, XT>(st, w, rows, cols, x, y); break;
        default: throw std::runtime_error("gemv_q8_planar: split-K supports 1..8 columns");
    }
}

template <typename XT>
void launch_split_mode(cudaStream_t st, const uint8_t * w, int64_t rows, int64_t cols, int64_t ncols,
                       const XT * x, float * y, int split) {
    switch (split) {
        case 1: launch_splitk_t<1, XT>(st, w, rows, cols, ncols, x, y); break;
        case 2:
            if ((cols / 2) % kStep) throw std::runtime_error("gemv_q8_planar: split2 needs cols % 512 == 0");
            launch_splitk_t<2, XT>(st, w, rows, cols, ncols, x, y);
            break;
        case 4:
            if ((cols / 4) % kStep) throw std::runtime_error("gemv_q8_planar: split4 needs cols % 1024 == 0");
            launch_splitk_t<4, XT>(st, w, rows, cols, ncols, x, y);
            break;
        default: throw std::runtime_error("gemv_q8_planar: split must be 1, 2 or 4");
    }
}

template <int R>
void launch(cudaStream_t st, const uint8_t * w, int64_t rows, int64_t cols, const float * x, float * y) {
    const int8_t * codes  = reinterpret_cast<const int8_t *>(w);
    const __half * scales = reinterpret_cast<const __half *>(w + rows * cols);
    const int64_t  rows_per_cta = (int64_t) kWarpsPerCta * R;
    const dim3     grid((unsigned) ((rows + rows_per_cta - 1) / rows_per_cta));
    const dim3     block(kWarp, kWarpsPerCta);
    gemv_q8_planar_k<R><<<grid, block, (size_t) cols * sizeof(float), st>>>(codes, scales, x, y, (int) rows,
                                                                          (int) cols);
}

// The fp16-activation split-K kernel reading GGUF's own interleaved Q8_0 layout -- no repack.
// A repack in place breaks prefill (T > 8 still runs ggml's MMQ, which needs GGUF layout), keeping both
// layouts costs ~1 GB of VRAM, and a planar repack of a fused span (qkv|z, q|k|v, shexp gate|up -- the
// model's real call sites) puts each tensor's scales after its own codes and breaks the span. GGUF's
// row-major 34-byte blocks stay contiguous across a span.
// The cost is the load: a block is 2 B scale + 32 B codes, so a lane's 8 codes at 34*blk + 2 + {0,8,16,24}
// are never 8-byte aligned -- but always 2-byte aligned, so they are four aligned 16-bit loads. The
// warp still reads one contiguous span (8 blocks = 272 B), so it stays bandwidth-bound.
template <int S, int T>
__global__ void gemv_q8_gguf_splitk_k(const uint8_t * __restrict__ w, const __half * __restrict__ x,
                                      float * __restrict__ y, int rows, int cols) {
    __shared__ float part[kWarpsPerCta][T];
    const int lane  = threadIdx.x;
    const int wp    = threadIdx.y;
    const int row   = blockIdx.x * (kWarpsPerCta / S) + wp / S;
    const int piece = wp % S;
    const int span  = cols / S;
    const int k0    = piece * span;
    const int nb    = cols >> 5;

    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        acc[t] = 0.0f;
    }
    if (row < rows) {
        const uint8_t * wrow = w + (size_t) row * nb * 34;
        for (int base = k0 + lane * 8; base < k0 + span; base += kStep) {
            const uint8_t *  blk = wrow + (size_t) (base >> 5) * 34;
            const float      d   = __half2float(*reinterpret_cast<const __half *>(blk));
            const uint16_t * cp  = reinterpret_cast<const uint16_t *>(blk + 2 + (base & 31));
            int2 v;
            v.x = (int) ((uint32_t) cp[0] | ((uint32_t) cp[1] << 16));
            v.y = (int) ((uint32_t) cp[2] | ((uint32_t) cp[3] << 16));
            const int8_t * c = reinterpret_cast<const int8_t *>(&v);
            float cw[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                cw[j] = (float) c[j];
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                float xv[8];
                load8(x + (size_t) t * cols + base, xv);
                float s = cw[0] * xv[0];
#pragma unroll
                for (int j = 1; j < 8; ++j) {
                    s = fmaf(cw[j], xv[j], s);
                }
                acc[t] = fmaf(d, s, acc[t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o, kWarp);
        }
    }
    if (S == 1) {
        if (lane == 0 && row < rows) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                y[(size_t) t * rows + row] = acc[t];
            }
        }
        return;
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            part[wp][t] = acc[t];
        }
    }
    __syncthreads();
    if (lane == 0 && piece == 0 && row < rows) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float sum = 0.0f;
#pragma unroll
            for (int p = 0; p < S; ++p) {
                sum += part[wp + p][t];
            }
            y[(size_t) t * rows + row] = sum;
        }
    }
}

template <int S, int T>
void launch_gguf(cudaStream_t st, const uint8_t * w, int64_t rows, int64_t cols, const __half * x, float * y) {
    const int64_t rows_per_cta = kWarpsPerCta / S;
    const dim3    grid((unsigned) ((rows + rows_per_cta - 1) / rows_per_cta));
    const dim3    block(kWarp, kWarpsPerCta);
    gemv_q8_gguf_splitk_k<S, T><<<grid, block, 0, st>>>(w, x, y, (int) rows, (int) cols);
}

template <int S>
void launch_gguf_t(cudaStream_t st, const uint8_t * w, int64_t rows, int64_t cols, int64_t ncols, const __half * x,
                   float * y) {
    switch (ncols) {
        case 1: launch_gguf<S, 1>(st, w, rows, cols, x, y); break;
        case 2: launch_gguf<S, 2>(st, w, rows, cols, x, y); break;
        case 3: launch_gguf<S, 3>(st, w, rows, cols, x, y); break;
        case 4: launch_gguf<S, 4>(st, w, rows, cols, x, y); break;
        case 5: launch_gguf<S, 5>(st, w, rows, cols, x, y); break;
        case 6: launch_gguf<S, 6>(st, w, rows, cols, x, y); break;
        case 7: launch_gguf<S, 7>(st, w, rows, cols, x, y); break;
        case 8: launch_gguf<S, 8>(st, w, rows, cols, x, y); break;
        default: throw std::runtime_error("gemv_q8_gguf_f16x: 1..8 columns");
    }
}

// f32 -> fp16, 4 elements per thread. A stand-in for a producer that emits fp16 itself: it keeps the
// launch count equal to the quantize_q8_1 it replaces. Folding this cast into the producers would
// remove the launch.
__global__ void f32_to_f16_k(const float * __restrict__ x, __half * __restrict__ y, int64_t n4) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        const float4 v = reinterpret_cast<const float4 *>(x)[i];
        __half2 * o = reinterpret_cast<__half2 *>(y) + 2 * i;
        o[0] = __floats2half2_rn(v.x, v.y);
        o[1] = __floats2half2_rn(v.z, v.w);
    }
}

}  // namespace

namespace ai::kern {

void gemv_q8_gguf_f16x(void * stream, const uint8_t * w, int64_t rows, int64_t cols, int64_t ncols,
                       const void * x_f16, float * y, int split) {
    if (cols % ((int64_t) split * kStep) != 0 || ncols < 1 || ncols > 8) {
        throw std::runtime_error("gemv_q8_gguf_f16x: cols % (split*256) == 0 and 1..8 columns");
    }
    const cudaStream_t st = (cudaStream_t) stream;
    const __half *     x  = (const __half *) x_f16;
    switch (split) {
        case 1: launch_gguf_t<1>(st, w, rows, cols, ncols, x, y); break;
        case 2: launch_gguf_t<2>(st, w, rows, cols, ncols, x, y); break;
        case 4: launch_gguf_t<4>(st, w, rows, cols, ncols, x, y); break;
        default: throw std::runtime_error("gemv_q8_gguf_f16x: split must be 1, 2 or 4");
    }
}

void cast_f32_f16(void * stream, const float * x, void * y, int64_t n) {
    if (n % 4 != 0) {
        throw std::runtime_error("cast_f32_f16: n must be a multiple of 4");
    }
    const int64_t n4 = n / 4;
    const int     bs = 256;
    f32_to_f16_k<<<(unsigned) ((n4 + bs - 1) / bs), bs, 0, (cudaStream_t) stream>>>(x, (__half *) y, n4);
}

void q8_repack_planar(const uint8_t * gguf, uint8_t * planar, int64_t rows, int64_t cols) {
    if (cols % kStep != 0) {
        throw std::runtime_error("q8_repack_planar: cols must be a multiple of 256");
    }
    const int64_t nb = cols / 32;
    uint8_t * codes  = planar;
    uint8_t * scales = planar + rows * cols;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t * blk = gguf + (r * nb + b) * 34;
            std::memcpy(scales + (r * nb + b) * 2, blk, 2);            // fp16 d
            std::memcpy(codes + r * cols + b * 32, blk + 2, 32);       // 32 x int8
        }
    }
}

bool gemv_q8_planar_ok(int64_t cols) {
    return cols % kStep == 0 && cols * (int64_t) sizeof(float) <= 48 * 1024;
}

void gemv_q8_planar(void * stream, const uint8_t * w, int64_t rows, int64_t cols, int64_t ncols, const float * x,
                    float * y, int mode) {
    if (!gemv_q8_planar_ok(cols)) {
        throw std::runtime_error("gemv_q8_planar: cols must be a multiple of 256 and fit shared memory");
    }
    if (ncols < 1 || ncols > 8) {
        throw std::runtime_error("gemv_q8_planar: 1..8 activation columns");
    }
    const cudaStream_t st = (cudaStream_t) stream;
    // mode 1/2/4: the shared-x kernel, rows per warp, one column only.
    // mode -1/-2/-4: split-K over 1/2/4 warps per row, 1..8 columns.
    if (mode > 0 && ncols != 1) {
        throw std::runtime_error("gemv_q8_planar: the shared-x kernel is single-column; use split-K for T > 1");
    }
    switch (mode) {
        case 1: launch<1>(st, w, rows, cols, x, y); break;
        case 2: launch<2>(st, w, rows, cols, x, y); break;
        case 4: launch<4>(st, w, rows, cols, x, y); break;
        case -1: case -2: case -4: launch_split_mode<float>(st, w, rows, cols, ncols, x, y, -mode); break;
        default: throw std::runtime_error("gemv_q8_planar: mode must be 1/2/4 or split -1/-2/-4");
    }
}

void gemv_q8_planar_f16x(void * stream, const uint8_t * w, int64_t rows, int64_t cols, int64_t ncols,
                         const void * x_f16, float * y, int split) {
    if (!gemv_q8_planar_ok(cols) || ncols < 1 || ncols > 8) {
        throw std::runtime_error("gemv_q8_planar_f16x: cols % 256 == 0 and 1..8 columns");
    }
    launch_split_mode<__half>((cudaStream_t) stream, w, rows, cols, ncols, (const __half *) x_f16, y, split);
}

bool gemv_q8_planar_split_ok(int64_t cols, int split) {
    return split >= 1 && cols % ((int64_t) split * kStep) == 0;
}

}  // namespace ai::kern
