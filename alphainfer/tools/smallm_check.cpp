// Kernel-level check for ALPHA_SMALLM_TC: ggml_mul_mat on CUDA for real 27B shapes at small M, dumped to a file.
// Run once with ALPHA_SMALLM_TC=0 (ggml's MMQ/MMVQ) and once with ALPHA_SMALLM_TC=1, then compare the dumps:
//   ai-smallm-check <out.bin> [--time]
//   ai-smallm-check --count <type> [--time]   column-count invariance on 35B dense shapes (q8_0, q6_K, ...): each
//       column's output at M = 1..8 must be bit-identical to the same column at M = 8. That is what exact MTP needs
//       when plain decode (M = 1) and the verify (M = k+1) run the same kernel (ALPHA_SMALLM_TC_MINCOLS=1).
// Both paths quantise activations to q8_1, so the integer products are identical and only the fp32 summation order
// differs: expect relative error ~1e-6, not 1e-3. --time also prints the mean kernel time per case (50 runs).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct Case { ggml_type type; int64_t K, rows, M; };

static int count_suite(ggml_backend_t be, ggml_type type, bool timing) {
    // 35B (hidden 2048) dense projection shapes: K x rows. AI_CHECK_27B=1: the 27B's (hidden 5120, ffn 17408) instead,
    // whose K-dependent split differs -- a count-invariance pass at K = 2048 says nothing about K = 5120 / 17408.
    static const int64_t s35[][2] = { {2048, 8192}, {2048, 4096}, {4096, 2048}, {2048, 512}, {512, 2048}, {2048, 12288}, {2048, 32768} };   // + an LM-head-like q6_K slice
    static const int64_t s27[][2] = { {5120, 17408}, {17408, 5120}, {5120, 6144}, {5120, 10240}, {6144, 5120}, {5120, 1024}, {5120, 32768} };
    const bool big = getenv("AI_CHECK_27B") && atoi(getenv("AI_CHECK_27B")) != 0;
    const int64_t (&shapes)[7][2] = big ? s27 : s35;
    const int     Ms[]        = { 1, 2, 3, 4, 5, 8 };
    int bad = 0;
    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const auto & sh : shapes) {
        const int64_t K = sh[0], rows = sh[1], MX = 8;
        if (K % ggml_blck_size(type) != 0) continue;
        std::vector<float> wf((size_t) (K * rows)), xf((size_t) (K * MX));
        for (auto & v : wf) v = nd(rng) * 0.02f;
        for (auto & v : xf) v = nd(rng);
        std::vector<uint8_t> wq(ggml_row_size(type, K) * (size_t) rows);
        ggml_quantize_chunk(type, wf.data(), wq.data(), 0, rows, K, nullptr);
        std::vector<std::vector<float>> outs;
        std::vector<double> tus;
        for (int M : Ms) {
            ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * W = ggml_new_tensor_2d(ctx, type, K, rows);
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            ggml_backend_tensor_set(W, wq.data(), 0, wq.size());
            ggml_backend_tensor_set(X, xf.data(), 0, (size_t) (K * M) * 4);   // the first M columns of the same X
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Y);
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out((size_t) (rows * M));
            ggml_backend_tensor_get(Y, out.data(), 0, out.size() * 4);
            outs.push_back(out);
            double us = 0;
            if (timing) {
                for (int i = 0; i < 5; ++i) ggml_backend_graph_compute(be, gf);
                ggml_backend_synchronize(be);
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 50; ++i) ggml_backend_graph_compute(be, gf);
                ggml_backend_synchronize(be);
                us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 50;
            }
            tus.push_back(us);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        // reference column values: M = 8 (the last arm); every column of every narrower arm must equal it bitwise
        const std::vector<float> & ref = outs.back();
        double num = 0, den = 0;
        {
            const auto * tr = ggml_get_type_traits(type);
            std::vector<float> wd((size_t) K);
            for (int64_t r = 0; r < rows; ++r) {
                tr->to_float(wq.data() + (size_t) r * ggml_row_size(type, K), wd.data(), K);
                double acc = 0;
                for (int64_t k = 0; k < K; ++k) acc += (double) wd[(size_t) k] * xf[(size_t) k];
                const double e = ref[(size_t) r] - acc;
                num += e * e;
                den += acc * acc;
            }
        }
        printf("%-6s K=%5lld rows=%6lld  rel-L2(col0 vs f64) %.3g |", ggml_type_name(type), (long long) K,
               (long long) rows, sqrt(num / den));
        for (size_t a = 0; a < outs.size(); ++a) {
            const int M = Ms[a];
            int64_t diff = 0;
            for (int m = 0; m < M; ++m) {
                diff += memcmp(outs[a].data() + (size_t) (m * rows), ref.data() + (size_t) (m * rows), (size_t) rows * 4) != 0;
            }
            bad += diff != 0;
            printf(" M=%d %s", M, diff ? "DIFF" : "same");
            if (timing) printf(" %.1fus", tus[a]);
        }
        printf("\n");
    }
    printf("COUNT-INVARIANCE: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out.bin> [--time]\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "--count") == 0) {
        if (argc < 3) return 2;
        ggml_type type = GGML_TYPE_COUNT;
        for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
            const char * n = ggml_type_name((ggml_type) t);
            if (n && strcmp(n, argv[2]) == 0) type = (ggml_type) t;
        }
        if (type == GGML_TYPE_COUNT) { fprintf(stderr, "unknown type %s\n", argv[2]); return 2; }
        ggml_backend_t be = ggml_backend_cuda_init(0);
        if (!be) return 1;
        const int rc = count_suite(be, type, argc > 3 && strcmp(argv[3], "--time") == 0);
        ggml_backend_free(be);
        return rc;
    }
    const bool timing = argc > 2 && strcmp(argv[2], "--time") == 0;
    const Case cases[] = {
        {GGML_TYPE_Q4_K, 5120, 17408, 8}, {GGML_TYPE_Q4_K, 17408, 5120, 8}, {GGML_TYPE_Q4_K, 5120, 6144, 7},
        {GGML_TYPE_Q5_K, 5120, 17408, 8}, {GGML_TYPE_Q5_K, 17408, 5120, 8}, {GGML_TYPE_Q5_K, 5120, 6144, 2},
        {GGML_TYPE_IQ4_XS, 5120, 17408, 8}, {GGML_TYPE_IQ4_XS, 17408, 5120, 8}, {GGML_TYPE_IQ4_XS, 5120, 6144, 7},
        {GGML_TYPE_Q6_K, 5120, 32768, 8}, {GGML_TYPE_Q6_K, 5120, 32768, 7}, {GGML_TYPE_Q6_K, 17408, 5120, 8},
    };
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) {
        fprintf(stderr, "cuda init failed\n");
        return 1;
    }
    FILE * fo = fopen(argv[1], "wb");
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const Case & c : cases) {
        // host data: weights quantised on the CPU reference path, activations f32
        std::vector<float> wf((size_t) (c.K * c.rows)), xf((size_t) (c.K * c.M));
        for (auto & v : wf) v = nd(rng) * 0.02f;
        for (auto & v : xf) v = nd(rng);
        std::vector<uint8_t> wq(ggml_row_size(c.type, c.K) * (size_t) c.rows);
        ggml_quantize_chunk(c.type, wf.data(), wq.data(), 0, c.rows, c.K, nullptr);

        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * W = ggml_new_tensor_2d(ctx, c.type, c.K, c.rows);
        ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.K, c.M);
        ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(W, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(X, xf.data(), 0, xf.size() * 4);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, Y);
        ggml_backend_graph_compute(be, gf);
        std::vector<float> out((size_t) (c.rows * c.M));
        ggml_backend_tensor_get(Y, out.data(), 0, out.size() * 4);
        fwrite(out.data(), 4, out.size(), fo);
        // exact reference: the dequantised weights times the original f32 activations, in f64 -- ground truth for
        // both kernels (each also quantises the activations, so neither is the reference for the other)
        {
            const auto * tr = ggml_get_type_traits(c.type);
            std::vector<float> wd((size_t) c.K);
            double num = 0, den = 0;
            for (int64_t r = 0; r < c.rows; ++r) {
                tr->to_float(wq.data() + (size_t) r * ggml_row_size(c.type, c.K), wd.data(), c.K);
                for (int64_t m = 0; m < c.M; ++m) {
                    double acc = 0;
                    const float * x = xf.data() + (size_t) m * c.K;
                    for (int64_t k = 0; k < c.K; ++k) acc += (double) wd[(size_t) k] * x[k];
                    const double e = out[(size_t) (m * c.rows + r)] - acc;
                    num += e * e;
                    den += acc * acc;
                }
            }
            printf("   vs exact f64 reference: rel-L2 %.3g" "\n", sqrt(num / den));
        }
        // Column-position invariance: rotate X's columns by one and recompute. A verify computes the same token at
        // different column indices in different rounds, so each column's output must be bit-identical in any slot.
        {
            std::vector<float> xr(xf.size());
            for (int64_t m = 0; m < c.M; ++m) {
                memcpy(xr.data() + (size_t) (((m + 1) % c.M) * c.K), xf.data() + (size_t) (m * c.K), (size_t) c.K * 4);
            }
            ggml_backend_tensor_set(X, xr.data(), 0, xr.size() * 4);
            ggml_backend_graph_compute(be, gf);
            std::vector<float> out2(out.size());
            ggml_backend_tensor_get(Y, out2.data(), 0, out2.size() * 4);
            int64_t diff = 0;
            for (int64_t m = 0; m < c.M; ++m) {
                diff += memcmp(out.data() + (size_t) (m * c.rows), out2.data() + (size_t) (((m + 1) % c.M) * c.rows),
                               (size_t) c.rows * 4) != 0;
            }
            printf("   column-rotation invariance: %lld of %lld columns differ" "\n", (long long) diff, (long long) c.M);
            ggml_backend_tensor_set(X, xf.data(), 0, xf.size() * 4);
        }
        double us = 0;
        if (timing) {
            for (int i = 0; i < 5; ++i) ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 50; ++i) ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 50;
        }
        printf("%-7s K=%6lld rows=%6lld M=%lld  out[0]=%+.6f  %s%.1f us\n", ggml_type_name(c.type), (long long) c.K,
               (long long) c.rows, (long long) c.M, out[0], timing ? "time " : "", us);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    fclose(fo);
    ggml_backend_free(be);
    return 0;
}
