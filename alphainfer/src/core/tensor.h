// SPDX-License-Identifier: Apache-2.0
// AlphaInfer core tensor view.
//
// A Tensor is a non-owning view: a device pointer, the buffer it lives in, a dtype and 4-D
// shape/strides. It carries no graph, no op and no ownership. Every Op takes and returns these.
//
// dtype ids mirror ggml_type values so the adapter Ops (which wrap ggml's kernels) can build
// descriptors without a translation table.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ai {

enum DType : int {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q8_0 = 8,
    Q2_K = 10,     // ggml GGML_TYPE_Q2_K: block 256, 84 B/block (draft head only)
    Q3_K = 11,     // ggml GGML_TYPE_Q3_K: block 256, 110 B/block (draft head only)
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    I8   = 24,
    I32  = 26,
    IQ4_NL = 20,   // ggml GGML_TYPE_IQ4_NL: block 32, 18 B/block
    IQ3_S  = 21,   // ggml GGML_TYPE_IQ3_S: block 256, 110 B/block (ggml-common.h static_assert)
    IQ4_XS = 23,   // ggml GGML_TYPE_IQ4_XS: block 256, 136 B/block
    I64  = 27,
    BF16 = 30,
    // GSQ-RCO (Flash-Next) expert types; sizes from ggml-common.h, checked against ggml_nbytes on every tensor
    // of the ISTA file by ai-q4x-inspect
    IQ2_XXS = 16,  // block 256, 66 B
    IQ2_XS  = 17,  // block 256, 74 B
    IQ3_XXS = 18,  // block 256, 98 B
    IQ2_S   = 22,  // block 256, 82 B
    Q2_0    = 45,  // block 64, 18 B (ggml id; ISTA files store it as 42, remapped by ggml's gguf.cpp)
    NVFP4 = 40,    // ggml GGML_TYPE_NVFP4: block 64, 36 B/block (4 UE4M3 sub-scales + 32 B E2M1 nibbles)
};

const char * dtype_name(int t);

struct Tensor {
    void *   data = nullptr;
    int      buf  = -1;          // Device buffer id the pointer lives in
    int      type = F32;
    int64_t  ne[4] = {1, 1, 1, 1};
    size_t   nb[4] = {0, 0, 0, 0};

    bool     valid() const { return data != nullptr; }
    int64_t  numel() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
    bool     contiguous() const;
    size_t   nbytes() const;     // bytes spanned, including block-quantised row sizes

    // Views. None of these touch memory; they only change pointer/shape/strides.
    Tensor   reshape(int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) const;
    Tensor   view(int64_t n0, int64_t n1, int64_t n2, int64_t n3,
                  size_t s1, size_t s2, size_t s3, size_t offset_bytes) const;
    Tensor   permute(int a0, int a1, int a2, int a3) const;   // same meaning as ggml_permute
    Tensor   transpose() const { return permute(1, 0, 2, 3); }

    std::string shape_str() const;
};

// Contiguous strides for (type, ne). Block-quantised types use the row size of ne[0].
void   set_contiguous_strides(Tensor & t);
size_t type_size(int type);      // bytes per block
int64_t block_size(int type);    // elements per block
size_t row_size(int type, int64_t n);

}  // namespace ai
