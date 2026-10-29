// SPDX-License-Identifier: Apache-2.0
#pragma once
// [AI_Q4X_FOLD_BATCH] every layer's speculative-commit fold in three launches (q4x_fold.cu)
#include <cstddef>
#include <cstdint>

namespace ai::kern {

struct FoldGdn {             // one GDN layer's record + committed state
    const float * co;        // conv_out f32 [C, T]: q | k | v per column
    const float * beta;      // raw beta, head stride sb1 / column stride sb2 (floats)
    const float * alpha;     // raw alpha (the gate before softplus), same strides
    int64_t       sb1, sb2;
    float *       ssm;       // f32 [Sv * Sv * Hv] state, advanced in place
    const float * dt;        // dt_bias [Hv]
    const float * a;         // ssm_a [Hv]
};
struct FoldCopy {            // dst[c * rows + j] = src[c * ld + m1 + j]
    const float * src;
    int64_t       ld;
    float *       dst;
    int           rows, cols;
};

size_t q4x_fold_scratch_floats(int n_gdn, int Sk, int Hk, int H, int tmax);
// false: a shape it does not take (the caller keeps the per-layer path)
bool q4x_fold_batch(void * stream, const FoldGdn * d_gdn, int n_gdn, const FoldCopy * d_cp, int n_cp, float * scratch, int m1,
                    int tmax, int C, int Sk, int Hk, int Sv, int Hv, float eps);

}  // namespace ai::kern
