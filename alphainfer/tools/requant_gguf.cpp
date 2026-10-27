// Requantise the 2-D matrices of a GGUF (any arch) to one ggml type, keeping every other tensor and all metadata:
//   ai-requant <in.gguf> <out.gguf> <type: q4_K|q5_K|q6_K|q8_0|q4_0> [skip-substring ...]
// A matrix is requantised when it is 2-D, its row length is a multiple of the target block, and its name contains
// none of the skip substrings. Source values come from the input's own dequantiser (ggml type traits), so start
// from the highest-precision artifact (e.g. the bf16 DFlash2 GGUF) rather than requantising a quantised one.
// Written for the DFlash2 drafter; keep "_cb" skipped there:
// the selector reads its codebooks as BF16.
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <in.gguf> <out.gguf> <q4_K|q5_K|q6_K|q8_0|q4_0> [skip-substring ...]\n", argv[0]);
        return 2;
    }
    const std::string ts = argv[3];
    const ggml_type   want = ts == "q4_K" ? GGML_TYPE_Q4_K : ts == "q5_K" ? GGML_TYPE_Q5_K
                           : ts == "q6_K" ? GGML_TYPE_Q6_K : ts == "q8_0" ? GGML_TYPE_Q8_0
                           : ts == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_COUNT;
    if (want == GGML_TYPE_COUNT) {
        fprintf(stderr, "unknown type %s\n", argv[3]);
        return 2;
    }
    std::vector<std::string> skip(argv + 4, argv + argc);

    ggml_context * src_data = nullptr;
    gguf_init_params ip = { /*no_alloc =*/ false, /*ctx =*/ &src_data };
    gguf_context * src = gguf_init_from_file(argv[1], ip);
    if (!src) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);

    const int64_t n = gguf_get_n_tensors(src);
    ggml_init_params cp = { ggml_tensor_overhead() * (size_t) (n + 1), nullptr, true };
    ggml_context * meta = ggml_init(cp);
    std::vector<std::vector<uint8_t>> keep;   // quantised payloads must outlive gguf_write_to_file
    keep.reserve((size_t) n);
    int64_t nq = 0;
    size_t  bytes_in = 0, bytes_out = 0;
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(src, i);
        ggml_tensor * t   = ggml_get_tensor(src_data, name);
        bool requant = ggml_n_dims(t) == 2 && t->ne[0] % ggml_blck_size(want) == 0 && t->type != want;
        for (const auto & sk : skip) {
            if (strstr(name, sk.c_str())) requant = false;
        }
        ggml_tensor * o = ggml_new_tensor(meta, requant ? want : t->type, ggml_n_dims(t), t->ne);
        ggml_set_name(o, name);
        bytes_in += ggml_nbytes(t);
        if (requant) {
            const auto * tr = ggml_get_type_traits(t->type);
            if (!tr->to_float) {
                fprintf(stderr, "%s: no dequantiser for %s\n", name, ggml_type_name(t->type));
                return 1;
            }
            std::vector<float> f((size_t) ggml_nelements(t));
            tr->to_float(t->data, f.data(), ggml_nelements(t));
            keep.emplace_back(ggml_row_size(want, t->ne[0]) * (size_t) t->ne[1]);
            ggml_quantize_chunk(want, f.data(), keep.back().data(), 0, t->ne[1], t->ne[0], nullptr);
            o->data = keep.back().data();
            ++nq;
            printf("  %-34s %-5s -> %-5s [%lld x %lld]\n", name, ggml_type_name(t->type), ggml_type_name(want),
                   (long long) t->ne[0], (long long) t->ne[1]);
        } else {
            o->data = t->data;
        }
        bytes_out += ggml_nbytes(o);
        gguf_add_tensor(dst, o);
    }
    if (!gguf_write_to_file(dst, argv[2], false)) {
        fprintf(stderr, "write failed\n");
        return 1;
    }
    printf("wrote %s: %lld tensors, %lld requantised to %s, %.2f -> %.2f GB\n", argv[2], (long long) n,
           (long long) nq, ggml_type_name(want), bytes_in / 1e9, bytes_out / 1e9);
    gguf_free(dst);
    gguf_free(src);
    ggml_free(meta);
    ggml_free(src_data);
    return 0;
}
