// SPDX-License-Identifier: Apache-2.0
// Per-stream sampling of the target token (temperature / top_k / top_p / min_p / seed).
//
// Runs after argmax_rows on the same rows and only rewrites rows whose stream has temperature > 0, so a greedy row
// keeps the argmax token bit for bit.
//
// A sampled row draws with the Gumbel-max trick: argmax_i (l_i / T + G_i), G_i = -log(-log(u_i)), over the tokens
// that survive the cuts. That is an exact draw from softmax(l / T) restricted to those tokens, over the whole
// vocabulary, at any temperature: no candidate list caps what can be drawn. The cuts become one logit threshold:
//   min_p  p_i >= min_p * p_max          <=>  l_i >= l_max + T * log(min_p)                         (exact)
//   top_k  the k largest logits           ->   l_i >= the k-th largest, k <= 64                     (k > 64: no cut)
//   top_p  the smallest prefix of the sorted distribution (after top_k) whose mass reaches top_p, found in the
//          64 largest; when the nucleus holds more than 64 tokens there is no top_p cut
// in the order the AlphaServe host sampler applies them (temperature, top_k, top_p, softmax), plus min_p.
//
// u_i comes from Philox4x32-10 keyed by the request seed, with counter (token id, position, slot): a draw depends
// only on (seed, position, slot, logits). A verify row and the plain decode of the same position therefore draw the
// same token from the same logits, so exact MTP verify stays token-identical to plain decode when sampling too.
#include "kernels/kernels.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

#define SMP_THREADS 1024   // threshold kernel: one block per row
#define SMP_TOPN    8      // per-thread candidate list; the block merges them into the 64 largest
#define SMP_CAND    64
#define GMB_THREADS 256    // Gumbel-max: argmax_rows' split
#define GMB_BLOCKS  256

namespace {

__device__ __forceinline__ uint32_t mulhilo(uint32_t a, uint32_t b, uint32_t & hi) {
    hi = __umulhi(a, b);
    return a * b;
}

// Philox4x32-10 (Salmon et al.), first output word.
__device__ __forceinline__ uint32_t philox(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0, uint32_t k1) {
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        uint32_t hi0, hi1;
        const uint32_t lo0 = mulhilo(0xD2511F53u, c0, hi0);
        const uint32_t lo1 = mulhilo(0xCD9E8D57u, c2, hi1);
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c0;
}

struct RowCtx {
    int   slot;
    long long pos;
};

__device__ __forceinline__ RowCtx row_ctx(long long row, long long L, int first_region, const int32_t * base,
                                          long long pos_const) {
    const long long b = row / L;
    RowCtx c;
    c.slot = first_region + (int) b;
    c.pos  = (base != nullptr ? (long long) base[b] + 1 : pos_const) + row % L;   // the position of the token picked
    return c;
}

}   // namespace

// thr[row] = the logit threshold, inv_t[row] = 1 / T (0 = greedy row, left alone by the later kernels).
__global__ static void sample_threshold_k(const float * __restrict__ x, long long ncols, long long L, int first_region,
                                          const int32_t * __restrict__ base, long long pos_const,
                                          const ai::kern::SampleParams * __restrict__ params,
                                          const int32_t * __restrict__ amax, float * __restrict__ thr,
                                          float * __restrict__ inv_t) {
    const long long row = blockIdx.x;
    const RowCtx    rc  = row_ctx(row, L, first_region, base, pos_const);
    const ai::kern::SampleParams p = params[rc.slot];
    if (!(p.temperature > 0.0f)) {
        if (threadIdx.x == 0) inv_t[row] = 0.0f;
        return;
    }
    const float * rowx = x + row * ncols;
    const float   it   = 1.0f / p.temperature;
    const float   lmax = rowx[amax[row]];
    const bool    want_k = p.top_k > 0 && p.top_k <= SMP_CAND && p.top_k < ncols;
    const bool    want_p = p.top_p > 0.0f && p.top_p < 1.0f;

    __shared__ float   s_red[32];
    __shared__ float   s_val[SMP_CAND];
    __shared__ int32_t s_win;
    __shared__ float   s_z;

    // pass 1: the partition function (for top_p without top_k) and each thread's SMP_TOPN largest
    float   tv[SMP_TOPN];
    int32_t ti[SMP_TOPN];
#pragma unroll
    for (int j = 0; j < SMP_TOPN; ++j) { tv[j] = -FLT_MAX; ti[j] = INT32_MAX; }
    float z = 0.0f;
    if (want_k || want_p) {
        for (long long col = threadIdx.x; col < ncols; col += SMP_THREADS) {
            const float v = rowx[col];
            if (want_p && !want_k) z += __expf((v - lmax) * it);
            if (v > tv[SMP_TOPN - 1]) {
                int j = SMP_TOPN - 1;
                while (j > 0 && v > tv[j - 1]) { tv[j] = tv[j - 1]; ti[j] = ti[j - 1]; --j; }
                tv[j] = v; ti[j] = (int32_t) col;
            }
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (want_p && !want_k) {
        for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
        if (lane == 0) s_red[warp] = z;
        __syncthreads();
        if (warp == 0) {
            float w = s_red[lane];
            for (int o = 16; o > 0; o >>= 1) w += __shfl_xor_sync(0xffffffffu, w, o);
            if (lane == 0) s_z = w;
        }
        __syncthreads();
    }
    // merge: SMP_CAND rounds of "largest head among all threads' sorted lists" (ties: lower token id)
    if (want_k || want_p) {
        __shared__ float   w_v[32];
        __shared__ int32_t w_i[32];
        int h = 0;
        for (int r = 0; r < SMP_CAND; ++r) {
            float   v = h < SMP_TOPN ? tv[h] : -FLT_MAX;
            int32_t i = h < SMP_TOPN ? ti[h] : INT32_MAX;
            for (int o = 16; o > 0; o >>= 1) {
                const float   ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int32_t oi = __shfl_xor_sync(0xffffffffu, i, o);
                if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
            }
            if (lane == 0) { w_v[warp] = v; w_i[warp] = i; }
            __syncthreads();
            if (warp == 0) {
                v = w_v[lane]; i = w_i[lane];
                for (int o = 16; o > 0; o >>= 1) {
                    const float   ov = __shfl_xor_sync(0xffffffffu, v, o);
                    const int32_t oi = __shfl_xor_sync(0xffffffffu, i, o);
                    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
                }
                if (lane == 0) { s_val[r] = v; s_win = i; }
            }
            __syncthreads();
            if (h < SMP_TOPN && ti[h] == s_win) ++h;   // token ids are unique: exactly one thread pops
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        float t = -FLT_MAX;
        if (p.min_p > 0.0f && p.min_p <= 1.0f) t = fmaxf(t, lmax + p.temperature * __logf(p.min_p));
        const int k = want_k ? p.top_k : SMP_CAND;
        if (want_k) t = fmaxf(t, s_val[k - 1]);
        if (want_p) {
            float zz = 0.0f;
            if (want_k) {
                for (int j = 0; j < k; ++j) zz += __expf((s_val[j] - lmax) * it);
            } else {
                zz = s_z;
            }
            float cum = 0.0f;
            for (int j = 0; j < k; ++j) {
                cum += __expf((s_val[j] - lmax) * it) / zz;
                if (cum >= p.top_p) { t = fmaxf(t, s_val[j]); break; }
            }
        }
        thr[row]   = t;
        inv_t[row] = it;
    }
}

__global__ static void sample_gumbel_partial_k(const float * __restrict__ x, long long ncols, long long L,
                                               int first_region, const int32_t * __restrict__ base, long long pos_const,
                                               const ai::kern::SampleParams * __restrict__ params,
                                               const float * __restrict__ thr, const float * __restrict__ inv_t,
                                               float * __restrict__ pmax, int32_t * __restrict__ pidx) {
    const long long row = blockIdx.y;
    const float     it  = inv_t[row];
    if (it == 0.0f) return;
    const RowCtx    rc  = row_ctx(row, L, first_region, base, pos_const);
    const uint64_t  seed = ((uint64_t) params[rc.slot].seed_hi << 32) | params[rc.slot].seed_lo;
    const uint32_t  k0 = (uint32_t) seed, k1 = (uint32_t) (seed >> 32);
    const uint32_t  p0 = (uint32_t) rc.pos, p1 = (uint32_t) ((unsigned long long) rc.pos >> 32);
    const float     t  = thr[row];
    const float *   rowx = x + row * ncols;

    __shared__ float   smax[GMB_THREADS];
    __shared__ int32_t sidx[GMB_THREADS];
    float   best = -FLT_MAX;
    int32_t bidx = -1;
    for (long long col = (long long) blockIdx.x * GMB_THREADS + threadIdx.x; col < ncols;
         col += (long long) GMB_BLOCKS * GMB_THREADS) {
        const float v = rowx[col];
        if (v < t) continue;
        const uint32_t r = philox((uint32_t) col, p0, p1, (uint32_t) rc.slot, k0, k1);
        const float    u = ((float) (r >> 8) + 0.5f) * (1.0f / 16777216.0f);   // (0, 1)
        const float    key = v * it - logf(-logf(u));
        if (key > best) { best = key; bidx = (int32_t) col; }
    }
    smax[threadIdx.x] = best;
    sidx[threadIdx.x] = bidx;
    __syncthreads();
    for (int st = GMB_THREADS / 2; st > 0; st >>= 1) {
        if ((int) threadIdx.x < st && smax[threadIdx.x + st] > smax[threadIdx.x]) {
            smax[threadIdx.x] = smax[threadIdx.x + st];
            sidx[threadIdx.x] = sidx[threadIdx.x + st];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pmax[row * GMB_BLOCKS + blockIdx.x] = smax[0];
        pidx[row * GMB_BLOCKS + blockIdx.x] = sidx[0];
    }
}

__global__ static void sample_gumbel_final_k(const float * __restrict__ inv_t, const float * __restrict__ pmax,
                                             const int32_t * __restrict__ pidx, int32_t * __restrict__ dst) {
    const long long row = blockIdx.x;
    if (inv_t[row] == 0.0f) return;
    __shared__ float   smax[GMB_BLOCKS];
    __shared__ int32_t sidx[GMB_BLOCKS];
    smax[threadIdx.x] = pmax[row * GMB_BLOCKS + threadIdx.x];
    sidx[threadIdx.x] = pidx[row * GMB_BLOCKS + threadIdx.x];
    __syncthreads();
    for (int st = GMB_BLOCKS / 2; st > 0; st >>= 1) {
        if ((int) threadIdx.x < st && smax[threadIdx.x + st] > smax[threadIdx.x]) {
            smax[threadIdx.x] = smax[threadIdx.x + st];
            sidx[threadIdx.x] = sidx[threadIdx.x + st];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && sidx[0] >= 0) dst[row] = sidx[0];   // the argmax token survives the cuts, so >= 0
}

namespace ai::kern {

int64_t sample_scratch_floats(int64_t rows) {
    return rows * (2 + 2 * (int64_t) GMB_BLOCKS);   // thr, inv_t, then pmax / pidx (an int32 per float slot)
}

void sample_rows(void * stream, const float * x, int64_t ncols, int64_t rows, int64_t L, int first_region,
                 const int32_t * base, int64_t pos_const, const SampleParams * params, float * scratch, int32_t * dst) {
    float *   thr   = scratch;
    float *   inv_t = scratch + rows;
    float *   pmax  = scratch + 2 * rows;
    int32_t * pidx  = (int32_t *) (scratch + 2 * rows + rows * GMB_BLOCKS);
    cudaStream_t s = (cudaStream_t) stream;
    sample_threshold_k<<<(unsigned) rows, SMP_THREADS, 0, s>>>(x, (long long) ncols, (long long) L, first_region, base,
                                                               (long long) pos_const, params, dst, thr, inv_t);
    sample_gumbel_partial_k<<<dim3(GMB_BLOCKS, (unsigned) rows), GMB_THREADS, 0, s>>>(
        x, (long long) ncols, (long long) L, first_region, base, (long long) pos_const, params, thr, inv_t, pmax, pidx);
    sample_gumbel_final_k<<<(unsigned) rows, GMB_BLOCKS, 0, s>>>(inv_t, pmax, pidx, dst);
}

}   // namespace ai::kern
