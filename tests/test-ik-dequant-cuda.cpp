// GPU-vs-CPU parity for the ik_llama.cpp dequant kernels.
// Fills a row per type (2 QK_K blocks + row-meta prefix, half scales normalized to 1.0),
// dequantizes on the CPU (ggml traits to_float) and on the GPU (convert-ik.cu launcher),
// and reports the max abs diff. GPU path is the exact scalar port, so diffs should be ~0
// (only sub-ULP FMA/half-rounding).
#include "ggml.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

extern "C" void ggml_cuda_dequantize_ik_rows_f32(const void * vx, void * y, int64_t nrows, int64_t ne, int64_t row_stride, int type, cudaStream_t stream);

static void set_half_one(std::vector<uint8_t>& b, size_t off) { if (off + 2 <= b.size()) { b[off] = 0x00; b[off+1] = 0x3c; } }

int main() {
    const enum ggml_type types[] = { GGML_TYPE_IQ4_K, GGML_TYPE_IQ6_K, GGML_TYPE_IQ4_KS,
                                     GGML_TYPE_IQ5_KS, GGML_TYPE_IQ3_KS, GGML_TYPE_IQ1_KT };
    const int64_t ne = 512;
    int fails = 0;
    for (size_t t = 0; t < sizeof(types)/sizeof(types[0]); ++t) {
        const enum ggml_type type = types[t];
        const struct ggml_type_traits * tr = ggml_get_type_traits(type);
        const size_t row_bytes = ggml_row_size(type, ne);
        std::vector<uint8_t> buf(row_bytes);
        uint32_t x = 0x12345678u;
        for (size_t i = 0; i < row_bytes; ++i) { x = x*1664525u + 1013904223u; buf[i] = (uint8_t)(x & 0xff); }
        if (type == GGML_TYPE_IQ4_K)  { set_half_one(buf, 0); set_half_one(buf, 144); }
        if (type == GGML_TYPE_IQ6_K)  { set_half_one(buf, 0); set_half_one(buf, 212); }
        if (type == GGML_TYPE_IQ3_KS) { set_half_one(buf, 0); }

        std::vector<float> cpu(ne, 0.f);
        tr->to_float(buf.data(), cpu.data(), ne);

        void * dx = nullptr; float * dy = nullptr;
        cudaMalloc(&dx, row_bytes); cudaMalloc(&dy, ne*sizeof(float));
        cudaMemcpy(dx, buf.data(), row_bytes, cudaMemcpyHostToDevice);
        ggml_cuda_dequantize_ik_rows_f32(dx, dy, 1, ne, (int64_t)row_bytes, (int)type, 0);
        cudaDeviceSynchronize();
        std::vector<float> gpu(ne, 0.f);
        cudaMemcpy(gpu.data(), dy, ne*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dx); cudaFree(dy);

        double maxabs = 0.0, maxrel = 0.0;
        for (int i = 0; i < ne; ++i) {
            double a = std::fabs((double)gpu[i] - (double)cpu[i]);
            double r = a / (std::fabs((double)cpu[i]) + 1e-9);
            if (a > maxabs) maxabs = a;
            if (r > maxrel) maxrel = r;
        }
        const bool ok = maxrel < 1e-3 || maxabs < 1e-3;
        printf("%-7s ne=%lld maxabs=%.3g maxrel=%.3g cpu[0]=%.6g gpu[0]=%.6g  %s\n",
               ggml_type_name(type), (long long)ne, maxabs, maxrel, cpu[0], gpu[0], ok ? "OK" : "FAIL");
        if (!ok) fails++;
    }
    return fails == 0 ? 0 : 1;
}
