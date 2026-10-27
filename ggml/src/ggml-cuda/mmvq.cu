#include "mmvq.cuh"
#include "dre.cuh"

// ALPHA_KQ_HOIST (default off): hoists the K-quant activation-side work (four u loads, two d8 loads, the
// integer dot2 sum) out of the rows_per_block loop, removing ~4 of the ~8 dp4a per call. Bit-identical,
// but it gives no measurable speedup: the q5_K MMVQ kernel is not ALU-bound at decode. Its bandwidth gap
// to q8_0 is more likely on the load side (q5_K reads three streams per block: qs, qh, packed scales).
#ifndef ALPHA_KQ_HOIST
#define ALPHA_KQ_HOIST 0
#endif
#include "q8stage.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>

// Batch-invariant decode: pin the MMVQ K-reduction warp split to the M=1 shape so a token's dot product
// reduces across the same number of warps regardless of batch width. Otherwise calc_nwarps returns 4 for
// M<=4 but 2 for M=5..8, so the same token gets a different partial-sum tree (and possibly a different
// argmax) depending on how many others it is batched with. The M columns are still batched.
//
// -DGGML_CUDA_BATCH_INVARIANT enables batch-invariant decode in MMVQ, MMQ and flash-attn and implies
// ALPHA_BATCH_INVARIANT_MMVQ; -DALPHA_BATCH_INVARIANT_MMVQ=1 also works on its own. Default off.
#if defined(GGML_CUDA_BATCH_INVARIANT) && !defined(ALPHA_BATCH_INVARIANT_MMVQ)
#define ALPHA_BATCH_INVARIANT_MMVQ 1
#endif
#ifndef ALPHA_BATCH_INVARIANT_MMVQ
#define ALPHA_BATCH_INVARIANT_MMVQ 0   // enable with -DALPHA_BATCH_INVARIANT_MMVQ=1 or -DGGML_CUDA_BATCH_INVARIANT
#endif

// Offset table lookup (defined in ggml-backend.cpp)
GGML_API_C void * ggml_backend_get_tensor_offsets(const void * tensor_ptr);

// Offset info struct (defined in ggml-cuda.cu)
struct pcache_offset_info {
    void *    pool_base;
    int64_t * d_offsets;
};

// Sentinel marking an expert slot as "not mine, skip it": the device-resolved expert tables (dre.cu,
// ALPHA_DRE_CPU=1) publish it for experts the CPU computes. Each expert slot writes its own slice of the
// [n_embd, n_expert_used, n_tokens] output, so a skipped slot leaves its slice to the other writer.
// A skipped slot returns before computing anything; this is safe against the __syncthreads() below
// because channel_x = ids[blockIdx.y] is block-uniform.
#define PCACHE_SLOT_SKIP INT64_MIN

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id) {
#if ALPHA_BATCH_INVARIANT_MMVQ
    ncols_dst = 1;  // pin warp-split to M=1 -> K-reduction identical for all batch widths
#endif
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        // ALPHA_MMVQ_NWARPS_LO / _HI override nwarps for ncols 2..4 / 5..8 (0 = stock). The stock
        // schedule cuts warps as ncols_dst grows, but nwarps also sets the K split (blocks_per_iter),
        // so at ncols 5..8 each thread covers twice as much of K and ncols times as many columns.
// ALPHA_MMVQ_CHAINS: independent accumulator chains per (column, row). `tmp[j][i] += ...` is one serial
// dependency chain across kbx iterations, which exposes dp4a latency at small N; rotating the chain by
// kbx gives the scheduler independent adds to interleave. 1 = stock.
#ifndef ALPHA_MMVQ_CHAINS
#define ALPHA_MMVQ_CHAINS 1
#endif
#ifndef ALPHA_MMVQ_NWARPS_LO
#define ALPHA_MMVQ_NWARPS_LO 0
#endif
#ifndef ALPHA_MMVQ_NWARPS_HI
#define ALPHA_MMVQ_NWARPS_HI 0
#endif
        switch (ncols_dst) {
            case 1:
                return 4;
            case 2:
            case 3:
            case 4:
                return ALPHA_MMVQ_NWARPS_LO > 0 ? ALPHA_MMVQ_NWARPS_LO : 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return ALPHA_MMVQ_NWARPS_HI > 0 ? ALPHA_MMVQ_NWARPS_HI : 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
#if ALPHA_BATCH_INVARIANT_MMVQ
    ncols_dst = 1;  // pin rows-per-block to the M=1 shape, matching the calc_nwarps pin, so the whole
                    // (nwarps, rows_per_block) reduction geometry is identical for every batch width
#endif
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

// small_k: 0 = off, 1 = small_k (rows_per_block = nwarps, the block's threads split K), 2 = small_k with one WARP
// per row (ALPHA_MMVQ_WARP_ROW, see the kernel)
template <ggml_type type, int ncols_dst, bool has_fusion, int small_k = 0, int wr_rows = 1>
__launch_bounds__((small_k == 2 ? calc_nwarps(type, ncols_dst, get_device_table_id()) : 1)*
                  calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        const int64_t * __restrict__ expert_offsets,  // pool block offsets for vx per expert (nullptr = use stride)
        const int64_t * __restrict__ gate_offsets,    // pool block offsets for fusion.gate per expert
        const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k != 0, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*(small_k == 2 ? wr_rows : 1)*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    sample_dst = blockIdx.z;

    // The ids row is selected by the sample index, so a multi-token MUL_MAT_ID can map its tokens onto the
    // sample dimension and still run this (ncols_dst == 1) arithmetic, bit-identical to per-token calls
    // (ALPHA_MMID_BI). With a single sample the added term is zero.
    channel_x  = ncols_dst == 1 && ids ? ids[sample_dst*ids_stride + channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
    }

    float x_biases[ncols_dst]    = { 0.0f };
    float gate_biases[ncols_dst] = { 0.0f };
    if constexpr (has_fusion) {
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias = x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            // 1. Hide latency by prefetching bias and gate here
            // 2. load only on threads that won't die after partial sum calculation
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
        if (use_gate_bias) {
            gate_bias = gate_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block][ALPHA_MMVQ_CHAINS] = {{{0.0f}}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    // Pool-direct reads: when the expert cache uploaded a per-expert block-offset table this pass
    // (all-hit MUL_MAT_ID tensor), vx points at the persistent GPU pool and expert_offsets[eid] gives the
    // expert's base in blocks. Each expert's data stays contiguous; only the base differs. The fused gate
    // needs its own table because the gate tensor lives at different pool offsets than the up tensor.
    // A table pointer with bit 0 set holds byte offsets (int64) from vx instead of block offsets
    // (ALPHA_DRE): device-resolved experts point each entry at a VRAM pool slot or at the expert's pinned
    // host copy, and those distances overflow an int block index. Tables are 8-byte aligned, so the tag
    // bit is otherwise zero. The same applies to the fused gate's table.
    const bool eo_bytes = ((uintptr_t) expert_offsets & 1) != 0;
    expert_offsets = (const int64_t *) ((uintptr_t) expert_offsets & ~(uintptr_t) 1);
    gate_offsets   = (const int64_t *) ((uintptr_t) gate_offsets   & ~(uintptr_t) 1);
    const bool use_offsets = ncols_dst == 1 && ids && expert_offsets != nullptr;
    // Skipped slot: it belongs to another writer, so neither compute nor store.
    // Block-uniform (channel_x comes from blockIdx.y), so returning here cannot strand a
    // subset of the block at the __syncthreads() further down.
    if (use_offsets && expert_offsets[channel_x] == PCACHE_SLOT_SKIP) {
        return;
    }
    // Bit 1 of a byte-offset entry marks an expert read zero-copy from its pinned host copy (expert
    // sizes are multiples of 4, so the bit is free); it is stripped before use
    bool zc = false;
    if (use_offsets && eo_bytes) {
        const int64_t off = expert_offsets[channel_x];
        zc = (off & 2) != 0;
        vx = (const char *) vx + (off & ~(int64_t) 3);
    }
    const int kbx_offset = use_offsets
        ? (eo_bytes ? 0 : (int)expert_offsets[channel_x]) + row0*stride_row_x
        : sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
    int gate_kbx_offset = kbx_offset;
    if constexpr (has_fusion) {
        if (use_gate && use_offsets) {
            // Host guarantees gate_offsets != nullptr whenever expert_offsets is set
            // on a fused launch (both tensors must have tables or neither).
            if (eo_bytes) {
                vgate = (const char *) vgate + (gate_offsets[channel_x] & ~(int64_t) 3);
                gate_kbx_offset = row0*stride_row_x;
            } else {
                gate_kbx_offset = (int)gate_offsets[channel_x] + row0*stride_row_x;
            }
        }
    }

    // small_k regrouped as warps per row (ALPHA_MMVQ_WARP_ROW). Plain small_k covers K in one pass of the
    // block (rows_per_block == nwarps, each thread at most one k-block) but gives every row to the same
    // threads: thread t computes k-slot t for all rows serially, and threads past the row's k-slots idle
    // (e.g. K = 640 uses 20 of 128 threads). Here the block is (warp_size, rows_per_block*wpr): the wpr warps
    // of row group i hold exactly the values threads 0..32*wpr-1 held for row i, summed in the same order
    // (missing warps contribute +0.0f, as idle threads did), then the same butterfly. Bit-identical, with one
    // vec_dot per thread instead of rows_per_block.
    if constexpr (small_k == 2 && ncols_dst == 1) {
        const int wpr = blockDim.y / rows_per_cuda_block;
        const int i   = threadIdx.y / wpr;
        const int sub = threadIdx.y % wpr;
        // The row group computes wr_rows rows (i, i + rows_per_block, ...) (ALPHA_MMVQ_WR_ROWS). Each row's k-slots,
        // vec_dots and reduction order match wr_rows == 1 (bit-identical); a block streams wr_rows x the bytes, so a
        // tiny expert row does not pay a whole block's latency on its own.
        float acc[wr_rows], acc_gate[wr_rows];
#pragma unroll
        for (int r = 0; r < wr_rows; ++r) { acc[r] = 0.0f; acc_gate[r] = 0.0f; }
        const int kbx = (sub*warp_size + threadIdx.x) / (qi/vdr);
        // Zero-copy expert: the block first copies its rows (and the fused gate's) from host memory into shared
        // memory with consecutive 32-bit loads, which coalesce into 128-byte PCIe reads, instead of the vec_dot's
        // scattered narrow reads. Only the staged types below get the 8 KB buffer; for others it would only cost
        // occupancy.
        constexpr int ZC_SMEM = (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_Q2_0) ? 8192 : 16;
        __shared__ __align__(16) uint32_t zc_buf[ZC_SMEM/4];
        const void * vxr = vx; const void * vgr = vgate;
        int xoff = kbx_offset, goff = gate_kbx_offset;
        if (zc) {
            constexpr size_t bsz = type == GGML_TYPE_IQ2_S ? sizeof(block_iq2_s) :
                                   type == GGML_TYPE_Q2_0  ? sizeof(block_q2_0)  : 0;   // staged types only
            const size_t row_bytes = (size_t) stride_row_x * bsz;
            const int nrow = min(rows_per_cuda_block*wr_rows, (int) stride_col_dst - row0);   // never past the last row
            const bool g2 = has_fusion && use_gate;
            const size_t need = row_bytes * nrow * (g2 ? 2 : 1);
            if (bsz > 0 && need <= ZC_SMEM && (row_bytes & 3) == 0) {
                const int tid_b = threadIdx.y*warp_size + threadIdx.x, nth = blockDim.y*warp_size;
                const uint32_t * src = (const uint32_t *) ((const char *) vx + (size_t) kbx_offset * bsz);
                const int nw = (int) (row_bytes * nrow / 4);
                for (int w = tid_b; w < nw; w += nth) zc_buf[w] = __ldcv(src + w);
                if (g2) {
                    const uint32_t * gsrc = (const uint32_t *) ((const char *) vgate + (size_t) gate_kbx_offset * bsz);
                    for (int w = tid_b; w < nw; w += nth) zc_buf[nw + w] = __ldcv(gsrc + w);
                }
                __syncthreads();
                vxr = zc_buf; xoff = 0;
                if (g2) { vgr = (const char *) zc_buf + row_bytes * nrow; goff = 0; }
            }
        }
        if (kbx < blocks_per_row_x) {
            const int kqs = vdr * ((sub*warp_size + threadIdx.x) % (qi/vdr));
            const int kby = kbx * (qk/QK8_1);
#pragma unroll
            for (int r = 0; r < wr_rows; ++r) {
                const int ir = r*rows_per_cuda_block + i;
                if (wr_rows == 1 || uint32_t(row0 + ir) < stride_col_dst) {
                    acc[r] = vec_dot_q_cuda(vxr, &y[kby], xoff + ir*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            acc_gate[r] = vec_dot_q_cuda(vgr, &y[kby], goff + ir*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
        // warp 0 of a row group keeps its own partial in registers: only warps 1..nwarps-1 need a slot
        constexpr int wr_np = nwarps > 1 ? nwarps - 1 : 1;
        __shared__ float wr_part[wr_rows][rows_per_cuda_block][wr_np][warp_size];
        __shared__ float wr_part_g[has_fusion ? wr_rows : 1][has_fusion ? rows_per_cuda_block : 1][has_fusion ? wr_np : 1][warp_size];
        if (sub > 0) {
#pragma unroll
            for (int r = 0; r < wr_rows; ++r) {
                wr_part[r][i][sub - 1][threadIdx.x] = acc[r];
                if constexpr (has_fusion) {
                    wr_part_g[r][i][sub - 1][threadIdx.x] = acc_gate[r];
                }
            }
        }
        __syncthreads();
        if (sub > 0) {
            return;
        }
#pragma unroll
        for (int r = 0; r < wr_rows; ++r) {
#pragma unroll
            for (int l = 1; l < nwarps; ++l) {
                acc[r] += l < wpr ? wr_part[r][i][l - 1][threadIdx.x] : 0.0f;
                if constexpr (has_fusion) {
                    if (use_gate) {
                        acc_gate[r] += l < wpr ? wr_part_g[r][i][l - 1][threadIdx.x] : 0.0f;
                    }
                }
            }
            acc[r] = warp_reduce_sum<warp_size>(acc[r]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    acc_gate[r] = warp_reduce_sum<warp_size>(acc_gate[r]);
                }
            }
            const int ir = r*rows_per_cuda_block + i;
            if (threadIdx.x == 0 && uint32_t(row0 + ir) < stride_col_dst) {
                float result = acc[r];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:     result *= ggml_cuda_op_silu_single(acc_gate[r]); break;
                            case GGML_GLU_OP_GEGLU:      result *= ggml_cuda_op_gelu_single(acc_gate[r]); break;
                            case GGML_GLU_OP_SWIGLU_OAI: result = ggml_cuda_op_swiglu_oai_single(acc_gate[r], result); break;
                            default:                     result = result * acc_gate[r]; break;
                        }
                    }
                }
                dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0 + ir] = result;
            }
        }
        GGML_UNUSED_VARS(x_biases, gate_biases, tmp, tmp_gate, blocks_per_iter, tid);
        return;
    }

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#ifndef ALPHA_MMVQ_XREUSE
#define ALPHA_MMVQ_XREUSE 0
#endif
        // ALPHA_MMVQ_XREUSE: dequantize x once per (row, k-block) and dot it against every column, instead of
        // re-deriving v[]/scales/mins/dm in each of the ncols_dst calls. Non-fused only: fusion is
        // ncols_dst == 1 only, which this branch never takes.
        constexpr bool xreuse_type = (type == GGML_TYPE_Q4_K) || (type == GGML_TYPE_Q5_K) ||
                                     (type == GGML_TYPE_Q6_K);
        if constexpr (ALPHA_MMVQ_XREUSE && xreuse_type && ncols_dst >= 2 && !has_fusion) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                float acc[ncols_dst] = {0.0f};
                const int xi = kbx_offset + i*stride_row_x + kbx;
                if constexpr (type == GGML_TYPE_Q4_K) {
                    vec_dot_q4_K_q8_1_multi<ncols_dst>(vx, &y[kby], stride_col_y, xi, kqs, acc);
                } else if constexpr (type == GGML_TYPE_Q5_K) {
                    vec_dot_q5_K_q8_1_multi<ncols_dst>(vx, &y[kby], stride_col_y, xi, kqs, acc);
                } else {
                    vec_dot_q6_K_q8_1_multi<ncols_dst>(vx, &y[kby], stride_col_y, xi, kqs, acc);
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) tmp[j][i][kbx % ALPHA_MMVQ_CHAINS] += acc[j];
            }
        } else {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i][kbx % ALPHA_MMVQ_CHAINS] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], gate_kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
        }
    }

    // collapse the independent accumulator chains before any reduction
    if constexpr (ALPHA_MMVQ_CHAINS > 1) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
                for (int c = 1; c < ALPHA_MMVQ_CHAINS; ++c) tmp[j][i][0] += tmp[j][i][c];
            }
        }
    }
    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    if constexpr (!has_fusion) {
        (void) tmp_shared_gate;
    } else if (!use_gate) {
        (void) tmp_shared_gate;
    }

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i][0];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i][0] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i][0] = warp_reduce_sum<warp_size>(tmp[j][i][0]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            float result = tmp[j][threadIdx.x][0];
            if constexpr (has_fusion) {
                if (use_bias) {
                    result += x_biases[j];
                }
                if (use_gate) {
                    float gate_value = tmp_gate[j][threadIdx.x];
                    if (use_gate_bias) {
                        gate_value += gate_biases[j];
                    }
                    switch (active_glu) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI: {
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        }
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            dst[j*stride_col_dst + threadIdx.x] = result;
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, gate_bias, x_bias, tmp_gate);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids,
        const int64_t * __restrict__ expert_offsets,  // pool block offsets per expert (nullptr = use stride)
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    // When expert_offsets is set: read directly from pool using per-expert block offsets.
    // expert_offsets[eid] = byte_offset / type_size (pre-computed on host).
    const int kbx_offset = expert_offsets
        ? (int)expert_offsets[channel_x] + row0*stride_row_x
        : channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};

    // ALPHA_KQ_HOIST: for the K-quants with a min term, the activation-side work (the four u loads, the two
    // d8 loads and the `dot2` sum) is identical for every row in this block, so compute it once per
    // activation block instead of once per row. Bit-identical: see kq_u_cache.
    constexpr bool kq_hoist = (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) && ALPHA_KQ_HOIST;

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

        if constexpr (kq_hoist) {
            kq_u_cache uc;
            const int bq8_offset = KQ_HOIST_QR * ((kqs/2) / (QI8_1/2));
            kq_u_cache_load(uc, &y[kby], bq8_offset, kqs);

#pragma unroll
            for (int i = 0; i < c_rows_per_block; ++i) {
                const int kbxi = kbx_offset + i*stride_row_x + kbx;
                if constexpr (type == GGML_TYPE_Q5_K) {
                    tmp[i] += vec_dot_q5_K_q8_1_hoisted(vx, uc, kbxi, kqs);
                } else {
                    tmp[i] += vec_dot_q4_K_q8_1_hoisted(vx, uc, kbxi, kqs);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < c_rows_per_block; ++i) {
                tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            }
        }
    }

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

// Row groups per warp group on the warp-row path (1, 2 or 4; bit-identical for every value).
// ALPHA_MMVQ_WR_ROWS=n forces n for every type; otherwise the default below.
static int mmvq_wr_rows(ggml_type type, int wpr, int nrows, int rpb) {
    static const int force = [] { const char * e = getenv("ALPHA_MMVQ_WR_ROWS"); return e ? atoi(e) : 0; }();
    // default 4: faster than 1 for the small IQ2/Q2_0/IQ4_NL expert shapes, and never slower
    int r = force > 0 ? force : 4;
    GGML_UNUSED(type); GGML_UNUSED(wpr);
    r = r >= 4 ? 4 : r >= 2 ? 2 : 1;
    while (r > 1 && nrows % (rpb * r) != 0) r /= 2;   // whole blocks only
    return r;
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, int small_k = 0, int wr_rows = 1>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids,
        const int64_t * expert_offsets, const int64_t * gate_offsets,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            mul_mat_vec_q<type, c_ncols_dst, true, small_k, wr_rows><<<block_nums, block_dims, nbytes_shared, stream>>>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    mul_mat_vec_q<type, c_ncols_dst, false, small_k, wr_rows><<<block_nums, block_dims, nbytes_shared, stream>>>
        (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids,
        const int64_t * expert_offsets,  // pool block offsets per expert (nullptr = use stride)
        const int64_t * gate_offsets,    // unused: the MoE multi-token kernel is never fused
        float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2;
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);

    GGML_UNUSED(gate_offsets);
    mul_mat_vec_q_moe<type, rows_per_block><<<block_nums, block_dims, 0, stream>>>(
        vx, vy, ids, expert_offsets, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids,
        const int64_t * expert_offsets,  // pool block offsets per expert (nullptr = use stride)
        const int64_t * gate_offsets,    // pool block offsets for the fused gate tensor
        const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_ids = ids != nullptr;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
        constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
        constexpr int vdr                   = get_vdr_mmvq(type);
        const int     blocks_per_row_x      = ncols_x / qk;
        const int     blocks_per_iter_1warp = vdr * warp_size / qi;
        const int     nwarps                = calc_nwarps(type, c_ncols_dst, table_id);
        bool          use                   = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        // ALPHA_MMVQ_IQ3_WARPROW (default 1) keeps IQ3_XXS / IQ3_S on the small-K path on Turing+: the exclusion
        // below predates the warp-row regrouping, with which small-K is ~15% faster for these types.
        static const bool iq3_wr = [] { const char * e = getenv("ALPHA_MMVQ_IQ3_WARPROW"); return !e || atoi(e) != 0; }();
        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 && !iq3_wr &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel (no fusion -> no gate table)
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, expert_offsets, gate_offsets, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;

            bool use_small_k = should_use_small_k(c_ncols_dst);

            // ALPHA_MMVQ_WARP_ROW (default on, bit-identical): small_k regrouped as wpr warps per row (see the kernel)
            static const bool warp_row_on = [] { const char * e = getenv("ALPHA_MMVQ_WARP_ROW"); return !e || atoi(e) != 0; }();
            constexpr int wr_qk = ggml_cuda_type_traits<type>::qk, wr_qi = ggml_cuda_type_traits<type>::qi;
            const int  wr_threads = (ncols_x / wr_qk) * (wr_qi / get_vdr_mmvq(type));   // k-slots per row
            const int  wr_wpr     = (wr_threads + warp_size - 1) / warp_size;
            const bool warp_row = use_small_k && warp_row_on && !fusion.x_bias && !fusion.gate_bias &&
                                  wr_wpr <= calc_nwarps(type, c_ncols_dst, table_id);
            if (warp_row) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id, true);
                dims.second.y *= wr_wpr;   // rows_per_block row groups of wr_wpr warps
                const int wr_rows = mmvq_wr_rows(type, wr_wpr, nrows_x, calc_rows_per_block(c_ncols_dst, table_id, true,
                                                 calc_nwarps(type, c_ncols_dst, table_id)));
                const int rpb_wr = calc_rows_per_block(c_ncols_dst, table_id, true, calc_nwarps(type, c_ncols_dst, table_id)) * wr_rows;
                dims.first.x = (nrows_x + rpb_wr - 1) / rpb_wr;
#define MMVQ_WR_LAUNCH(R) mul_mat_vec_q_switch_fusion<type, c_ncols_dst, 2, R>( \
                    vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst, \
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd, \
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride, stream)
                switch (wr_rows) {
                    case 4:  MMVQ_WR_LAUNCH(4); break;
                    case 2:  MMVQ_WR_LAUNCH(2); break;
                    default: MMVQ_WR_LAUNCH(1); break;
                }
#undef MMVQ_WR_LAUNCH
            } else if (use_small_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, 1>(
                    vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            } else {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(
                    vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

    GGML_UNUSED(has_fusion);
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids,
        const int64_t * expert_offsets,  // pool block offsets per expert (nullptr = use stride)
        const int64_t * gate_offsets,    // pool block offsets for the fused gate tensor
        const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, expert_offsets, gate_offsets, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// q8_1 activation cache, see mmvq.cuh. One entry, one persistent 8 MB device buffer (the pointer is baked into
// captured CUDA graphs, so it must never move); shapes above that fall back to the per-call pool path.
// ALPHA_Q8_CACHE=0 disables, ALPHA_Q8_CACHE_STATS=1 prints hit/miss/invalidate counts every 4096 graphs.
struct alpha_q8_cache {
    char *       buf        = nullptr;
    size_t       cap        = 0;
    bool         alloc_fail = false;
    const void * src1_data  = nullptr;
    size_t       src1_bytes = 0;
    int64_t      ne10 = 0, ne11 = 0, ne12 = 0, ne13 = 0, s11 = 0, s12 = 0, s13 = 0;
    bool         valid = false;
    long long    hits = 0, misses = 0, invalidations = 0, graphs = 0;
};
static alpha_q8_cache g_q8c;

static bool alpha_q8_cache_on() {
    static const bool v = []() {
        const char * e = getenv("ALPHA_Q8_CACHE");
        return e ? atoi(e) != 0 : true;
    }();
    return v;
}

void ggml_cuda_mmvq_q8_cache_begin_graph() {
    g_q8c.valid = false;
    g_q8c.graphs++;
    static const long long stats = []() {   // ALPHA_Q8_CACHE_STATS=N: print every N graphs (0/unset = off)
        const char * e = getenv("ALPHA_Q8_CACHE_STATS");
        const long long v = e ? atoll(e) : 0;
        return v == 1 ? 4096 : v;
    }();
    if (stats > 0 && g_q8c.graphs % stats == 0) {
        fprintf(stderr, "[q8-cache] graphs=%lld hits=%lld misses=%lld invalidations=%lld\n",
                g_q8c.graphs, g_q8c.hits, g_q8c.misses, g_q8c.invalidations);
    }
}

void ggml_cuda_mmvq_q8_cache_note_write(const void * data, size_t nbytes) {
    if (!g_q8c.valid || data == nullptr || nbytes == 0) {
        return;
    }
    const char * a = (const char *) data;
    const char * b = (const char *) g_q8c.src1_data;
    if (a < b + g_q8c.src1_bytes && b < a + nbytes) {
        g_q8c.valid = false;
        g_q8c.invalidations++;
    }
}

// ALPHA_MMID_SPLIT: largest token count whose MUL_MAT_ID is run as per-token 1-token calls. Keeps an MTP
// verify on the same kernel a decode token uses. Default 16 when ALPHA_MTP is on, 0 (stock) otherwise: for
// plain co-batched decode the per-token calls would re-read experts shared across users.
int alpha_mmid_split_max() {
    static const int v = []() {
        const char * e = getenv("ALPHA_MMID_SPLIT");
        return e ? atoi(e) : []() { const char * m = getenv("ALPHA_MTP"); return (m && atoi(m) != 0) ? 16 : 0; }();
    }();
    return v;
}

// Batch-invariant MUL_MAT_ID (ALPHA_MMID_BI, default on, 0 disables): instead of one 1-token call per token,
// run one call with the tokens on the kernel's sample axis. ncols_dst stays 1, so the warp split,
// rows-per-block and K-loop order match a decode token and the result is bit-identical to the split, but it
// is a single launch and experts shared by consecutive tokens are read while still in L2. It only replaces
// the ALPHA_MMID_SPLIT path.
static bool alpha_mmid_bi_on() {
    static const bool v = []() {
        const char * e = getenv("ALPHA_MMID_BI");
        return e ? atoi(e) != 0 : true;
    }();
    return v;
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, bool mmid_tokens_as_samples) {
    // Device-resolved experts, CPU miss path: ring the layer's doorbell once before its gate/up and wait for the answer
    // once after its down, around however the call below splits the tokens.
    {
        static thread_local bool dre_inner = false;
        if (!dre_inner && ids && ggml_cuda_dre_enabled()) {
            const void * b = nullptr; const int64_t * t = nullptr;
            if (ggml_cuda_dre_lookup(src0, nullptr, ctx.stream(), &b, &t)) {
                ggml_cuda_dre_pre(src0, src1, ids, ctx.stream());
                dre_inner = true;
                ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst, fusion, mmid_tokens_as_samples);
                dre_inner = false;
                ggml_cuda_dre_post(src0, dst, ctx.stream());
                return;
            }
        }
    }
    // The multi-token MoE kernel reduces K in a different order than the 1-token kernel, so a verify row
    // and a decode token of the same input disagree in the last bits. Run small batches token by token
    // instead, or with ALPHA_MMID_BI as one sample-axis call.
    if (!mmid_tokens_as_samples &&
        ids && src1->ne[2] >= 2 && src1->ne[2] <= alpha_mmid_split_max() && src1->ne[3] == 1 &&
        dst->ne[2] == src1->ne[2] && ids->ne[1] == src1->ne[2]) {
        // ne12 becomes one launch's sample count, so it has to stay inside the MMVQ batch bound.
        if (alpha_mmid_bi_on() && src1->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
            static bool logged = false;
            if (!logged) {
                logged = true;
                fprintf(stderr, "[mmid-bi] MUL_MAT_ID: %d tokens on the sample axis, one launch (ALPHA_MMID_BI)\n",
                        (int) src1->ne[2]);
            }
            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst, fusion, /* mmid_tokens_as_samples = */ true);
            return;
        }
        for (int64_t t = 0; t < src1->ne[2]; ++t) {
            ggml_tensor s1 = *src1; s1.ne[2] = 1; s1.data = (char *) src1->data + t * src1->nb[2];
            ggml_tensor iv = *ids;  iv.ne[1] = 1; iv.data = (char *) ids->data  + t * ids->nb[1];
            ggml_tensor dv = *dst;  dv.ne[2] = 1; dv.data = (char *) dst->data  + t * dst->nb[2];
            ggml_cuda_mul_mat_vec_q(ctx, src0, &s1, &iv, &dv, fusion);
        }
        return;
    }
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1 || mmid_tokens_as_samples);   // or tokens are on the sample axis
        GGML_ASSERT(  ids || dst->ne[1] == 1);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_bytes    = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    const char * src1_q8 = nullptr;
    // If the producer already emitted a q8_1 copy of this activation, use it and skip the quantize launch.
    // The geometry must match exactly: a partial match would make the GEMV silently read a different layout.
    if (fusion && fusion->src1_q8.data != nullptr &&
        fusion->src1_q8.ne10_padded == ne10_padded &&
        fusion->src1_q8.ne11 == ne11 && fusion->src1_q8.ne12 == ne12 && fusion->src1_q8.ne13 == ne13) {
        src1_q8 = (const char *) fusion->src1_q8.data;
        ggml_cuda_q8_stage_note_hit();
    }
    if (src1_q8 == nullptr) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        // Second route to the same thing: a ggml-node producer registers its staged copy by activation
        // pointer, so every GEMV benefits without threading fusion args through call sites. The fusion-args
        // field above remains the interface for producers with no ggml node.
        if (ggml_cuda_q8_stage_enabled()) {
            const char * staged = ggml_cuda_q8_stage_get(src1->data, ne10_padded, ne11, ne12, ne13, s11, s12, s13);
            if (staged != nullptr) {
                src1_q8 = staged;
                ggml_cuda_q8_stage_note_hit();
                ggml_cuda_q8_stage_note_hit_named(dst->name);
            }
        }
    }
    if (src1_q8 == nullptr) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        // The quantize is type-independent, so any GEMV (MUL_MAT or MUL_MAT_ID) over the
        // same src1 bytes with the same shape/strides can reuse the copy made by the previous one.
        constexpr size_t q8_cap = (size_t) 8 * 1024 * 1024;
        // [AI_Q8C_STREAM0] g_q8c is ONE global buffer: a GEMV on a second stream (AI_MOE_CONCURRENT) would
        // race stream 0 on it. Only stream 0 may use it; other streams take the per-stream pool path.
        if (alpha_q8_cache_on() && ctx.curr_stream_no == 0 && q8_bytes <= q8_cap && !g_q8c.alloc_fail) {
            if (g_q8c.buf == nullptr) {
                void * b = nullptr;
                if (cudaMalloc(&b, q8_cap) == cudaSuccess) {
                    g_q8c.buf = (char *) b;
                    g_q8c.cap = q8_cap;
                } else {
                    (void) cudaGetLastError();
                    g_q8c.alloc_fail = true;
                }
            }
            if (g_q8c.buf != nullptr) {
                const bool hit = g_q8c.valid && g_q8c.src1_data == src1->data &&
                    g_q8c.ne10 == ne10 && g_q8c.ne11 == ne11 && g_q8c.ne12 == ne12 && g_q8c.ne13 == ne13 &&
                    g_q8c.s11 == s11 && g_q8c.s12 == s12 && g_q8c.s13 == s13;
                if (hit) {
                    g_q8c.hits++;
                } else {
                    if (ggml_cuda_q8_stage_enabled()) ggml_cuda_q8_stage_note_miss(ne10_padded, ne11, ne12, ne13);
                    quantize_row_q8_1_cuda(src1_d, nullptr, g_q8c.buf, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
                    g_q8c.misses++;
                    g_q8c.src1_data = src1->data; g_q8c.src1_bytes = ggml_nbytes(src1);
                    g_q8c.ne10 = ne10; g_q8c.ne11 = ne11; g_q8c.ne12 = ne12; g_q8c.ne13 = ne13;
                    g_q8c.s11 = s11; g_q8c.s12 = s12; g_q8c.s13 = s13;
                    g_q8c.valid = true;
                }
                src1_q8 = g_q8c.buf;
            }
        }
        if (src1_q8 == nullptr) {
            if (ggml_cuda_q8_stage_enabled()) ggml_cuda_q8_stage_note_miss(ne10_padded, ne11, ne12, ne13);
            src1_q8_1.alloc(q8_bytes);
            quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
            src1_q8 = src1_q8_1.get();
        }
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // Tokens on the sample axis: one column, n_expert_used channels, ne12 samples. stride_sample_x
    // is 0 because the expert weights do not vary per token (the ids lookup already selects them), and
    // nsamples_x = 1 makes the kernel's sample_x zero for every token.
    int64_t bi_ncols_dst = 0, bi_nsamples_x = 0, bi_nsamples_dst = 0;
    int64_t bi_stride_sample_x = 0, bi_stride_sample_y = 0, bi_stride_sample_dst = 0;
    if (mmid_tokens_as_samples) {
        GGML_ASSERT(ids && ne3 == 1 && ne13 == 1 && dst->ne[2] == ne12 && ids->ne[1] == ne12);
        bi_ncols_dst         = 1;
        bi_nsamples_x        = 1;
        bi_nsamples_dst      = ne12;
        bi_stride_sample_x   = 0;
        bi_stride_sample_y   = ne11*s11;   // one token's quantised rows
        bi_stride_sample_dst = s2;         // one token's output plane
    }

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = mmid_tokens_as_samples ? bi_ncols_dst : (ids ? ne2 : ne1);
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    // Expert offset table: pool-direct MMVQ reads for all-hit MUL_MAT_ID tensors.
    // The pcache flush uploads a per-expert block-offset table when every used expert
    // of this tensor is pool-resident this pass; the map entry is per forward pass
    // (cleared in sched), so a stale table cannot be used. With vx = pool, expert_id*stride
    // addressing would read the wrong experts, so the kernel uses the per-expert table instead.
    // Fused gate/up: the gate tensor lives at different pool offsets, so both tensors need
    // their own table; if either lacks one, fall back to input_cpy reads (the flush D2D still populates it).
    const int64_t * expert_offsets = nullptr;
    const int64_t * gate_offsets   = nullptr;
    const void * vx_eff = src0->data;
    static const bool offset_table_disabled = getenv("ALPHA_NO_OFFSET_TABLE") != nullptr;
    // device-resolved experts: tagged byte-offset tables from a host base (dre.cu)
    bool dre_done = false;
    if (ids_d && ncols_dst == 1 && ggml_cuda_dre_enabled()) {   // the sample-axis form reads the table per token
        const void * b0 = nullptr; const int64_t * t0 = nullptr;
        if (ggml_cuda_dre_lookup(src0, ids, stream, &b0, &t0)) {
            if (fusion_local.gate) {
                const void * bg = nullptr; const int64_t * tg = nullptr;
                GGML_ASSERT(ggml_cuda_dre_lookup(fusion->gate, ids, stream, &bg, &tg) &&
                            "DRE: fused gate/up pair with only one tensor device-resolved");
                gate_offsets      = tg;
                fusion_local.gate = bg;
            }
            expert_offsets = t0;
            vx_eff         = b0;
            dre_done       = true;
        }
    }
    if (!dre_done && ids_d && ncols_dst == 1 && !mmid_tokens_as_samples && !offset_table_disabled) {
        pcache_offset_info * info = (pcache_offset_info *) ggml_backend_get_tensor_offsets(src0);
        if (info && info->d_offsets && info->pool_base) {
            if (fusion_local.gate) {
                pcache_offset_info * ginfo = (pcache_offset_info *) ggml_backend_get_tensor_offsets(fusion->gate);
                if (ginfo && ginfo->d_offsets && ginfo->pool_base) {
                    expert_offsets    = info->d_offsets;
                    gate_offsets      = ginfo->d_offsets;
                    vx_eff            = info->pool_base;
                    fusion_local.gate = ginfo->pool_base;
                }
            } else {
                expert_offsets = info->d_offsets;
                vx_eff         = info->pool_base;
            }
        }
    }

    mul_mat_vec_q_switch_type(
        vx_eff, src0->type, src1_q8, ids_d, expert_offsets, gate_offsets, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        mmid_tokens_as_samples ? bi_nsamples_x        : ne03,
        mmid_tokens_as_samples ? bi_nsamples_dst      : ne3,
        mmid_tokens_as_samples ? bi_stride_sample_x   : s03,
        mmid_tokens_as_samples ? bi_stride_sample_y   : s13,
        mmid_tokens_as_samples ? bi_stride_sample_dst : s3,
        ids_stride, stream);
    GGML_UNUSED(dre_done);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, nullptr, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
