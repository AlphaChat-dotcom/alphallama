// Golden harness for the ik_llama.cpp quant types ported into this ggml
// (IQ4_K, IQ6_K, IQ4_KS, IQ5_KS, IQ3_KS, IQ1_KT).
//
// For each type it fills one row of 512 elements (2 QK_K blocks, plus the
// per-row meta prefix where the type has one) with deterministic LCG bytes,
// dequantizes it via the type traits' to_float, and prints the first/last
// 4 floats plus a 32-bit FNV-1a hash of all 512 output floats.
//
// The same harness adapted to the ik_llama.cpp API (using ggml_internal_get_type_traits)
// can be built against ik's libggml on the same machine; matching hashes prove the port
// is bit-exact.
//
// Reference implementation ported from ik_llama.cpp
// (https://github.com/ikawrakow/ik_llama.cpp), MIT license, Copyright (c) Iwan Kawrakow.

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t fnv1a_32(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

int main() {
    const enum ggml_type types[] = {
        GGML_TYPE_IQ4_K,
        GGML_TYPE_IQ6_K,
        GGML_TYPE_IQ4_KS,
        GGML_TYPE_IQ5_KS,
        GGML_TYPE_IQ3_KS,
        GGML_TYPE_IQ1_KT,
    };

    const int64_t n_per_row = 512; // 2 blocks of QK_K = 256

    int n_fail = 0;

    for (size_t t = 0; t < sizeof(types)/sizeof(types[0]); ++t) {
        const enum ggml_type type = types[t];
        const struct ggml_type_traits * traits = ggml_get_type_traits(type);

        if (!traits->to_float) {
            printf("%-7s: MISSING to_float\n", ggml_type_name(type));
            n_fail++;
            continue;
        }

        // row bytes = row meta prefix + 2 blocks (ggml_row_size includes both)
        const size_t row_bytes = ggml_row_size(type, n_per_row);

        // deterministic fill: LCG re-seeded per type, take the low byte
        std::vector<uint8_t> buf(row_bytes);
        uint32_t x = 0x12345678u;
        for (size_t i = 0; i < row_bytes; ++i) {
            x = x*1664525u + 1013904223u;
            buf[i] = (uint8_t)(x & 0xff);
        }

        std::vector<float> out(n_per_row, 0.0f);
        traits->to_float(buf.data(), out.data(), n_per_row);

        const uint32_t h = fnv1a_32(out.data(), n_per_row*sizeof(float));

        printf("%-7s row_bytes=%4zu first[% .6g % .6g % .6g % .6g] last[% .6g % .6g % .6g % .6g] fnv1a=0x%08x\n",
               ggml_type_name(type), row_bytes,
               out[0], out[1], out[2], out[3],
               out[n_per_row-4], out[n_per_row-3], out[n_per_row-2], out[n_per_row-1],
               h);
    }

    return n_fail == 0 ? 0 : 1;
}
