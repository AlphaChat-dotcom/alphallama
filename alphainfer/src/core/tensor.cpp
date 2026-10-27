#include "core/tensor.h"

#include <cstdio>
#include <stdexcept>

namespace ai {

const char * dtype_name(int t) {
    switch (t) {
        case F32:  return "f32";
        case F16:  return "f16";
        case Q4_0: return "q4_0";
        case Q4_1: return "q4_1";
        case Q8_0: return "q8_0";
        case Q4_K: return "q4_K";
        case Q5_K: return "q5_K";
        case Q6_K: return "q6_K";
        case IQ4_NL: return "iq4_nl";
        case IQ4_XS: return "iq4_xs";
        case I8:   return "i8";
        case I32:  return "i32";
        case I64:  return "i64";
        case BF16: return "bf16";
        case IQ3_S: return "iq3_s";
        case NVFP4: return "nvfp4";
        case IQ2_XXS: return "iq2_xxs";
        case IQ2_XS: return "iq2_xs";
        case IQ3_XXS: return "iq3_xxs";
        case IQ2_S: return "iq2_s";
        case Q2_0: return "q2_0";
        default:   return "?";
    }
}

size_t type_size(int type) {
    switch (type) {
        case F32:  return 4;
        case F16:  return 2;
        case Q4_0: return 18;
        case Q4_1: return 20;
        case Q5_0: return 22;
        case Q8_0: return 34;
        case Q2_K: return 84;    // ggml-common.h: QK_K/16 + QK_K/4 + 2*sizeof(ggml_half)
        case Q3_K: return 110;   // ggml-common.h: QK_K/8 + QK_K/4 + 12 + sizeof(ggml_half)
        case Q4_K: return 144;
        case Q5_K: return 176;
        case Q6_K: return 210;
        // ggml-common.h:527/535 static_asserts are the source for both:
        //   block_iq4_nl = ggml_half + QK4_NL/2                        =   2 + 16 =  18
        //   block_iq4_xs = ggml_half + uint16 + QK_K/64 + QK_K/2       = 2+2+4+128 = 136
        case IQ4_NL: return 18;
        case IQ4_XS: return 136;
        case IQ3_S:  return 110;   // ggml-common.h: sizeof(ggml_half) + 13*(QK_K/32) + QK_K/64
        case I8:   return 1;
        case I32:  return 4;
        case I64:  return 8;
        case BF16: return 2;
        case NVFP4: return 36;     // ggml-common.h:245-251: QK_NVFP4/QK_NVFP4_SUB scales + QK_NVFP4/2 nibbles
        case IQ2_XXS: return 66;
        case IQ2_XS:  return 74;
        case IQ3_XXS: return 98;
        case IQ2_S:   return 82;
        case Q2_0:    return 18;
        default:   throw std::runtime_error("ai::type_size: unsupported dtype " + std::to_string(type));
    }
}

int64_t block_size(int type) {
    switch (type) {
        case Q4_0: case Q4_1: case Q5_0: case Q8_0: case IQ4_NL: return 32;
        case Q2_K: case Q3_K: case Q4_K: case Q5_K: case Q6_K: case IQ4_XS: case IQ3_S: return 256;
        case IQ2_XXS: case IQ2_XS: case IQ3_XXS: case IQ2_S: return 256;
        case NVFP4: case Q2_0: return 64;
        // unblocked scalar types
        case F32: case F16: case BF16: case I8: case I32: case I64: return 1;
        // Throws rather than returning 1: a default of 1 for an unknown blocked type would be a wrong
        // answer, and kv.cpp's alignment guard consumes block_size directly, so it would silently
        // accept a bad stride.
        default:
            throw std::runtime_error("ai::block_size: unsupported dtype " + std::to_string(type));
    }
}

size_t row_size(int type, int64_t n) {
    const int64_t bs = block_size(type);
    if (n % bs != 0) {
        throw std::runtime_error("ai::row_size: row length not a multiple of the block size");
    }
    return type_size(type) * (size_t) (n / bs);
}

void set_contiguous_strides(Tensor & t) {
    t.nb[0] = type_size(t.type);
    t.nb[1] = row_size(t.type, t.ne[0]);
    t.nb[2] = t.nb[1] * (size_t) t.ne[1];
    t.nb[3] = t.nb[2] * (size_t) t.ne[2];
}

bool Tensor::contiguous() const {
    Tensor c = *this;
    set_contiguous_strides(c);
    for (int i = 1; i < 4; ++i) {
        if (ne[i] > 1 && nb[i] != c.nb[i]) {
            return false;
        }
    }
    return ne[0] <= 1 || nb[0] == c.nb[0];
}

size_t Tensor::nbytes() const {
    if (numel() == 0) {
        return 0;
    }
    // same rule as ggml_nbytes
    const int64_t bs = block_size(type);
    size_t n;
    if (bs == 1) {
        n = type_size(type);
        for (int i = 0; i < 4; ++i) {
            n += (size_t) (ne[i] - 1) * nb[i];
        }
    } else {
        n = (size_t) ne[0] * nb[0] / (size_t) bs;
        for (int i = 1; i < 4; ++i) {
            n += (size_t) (ne[i] - 1) * nb[i];
        }
    }
    return n;
}

Tensor Tensor::reshape(int64_t n0, int64_t n1, int64_t n2, int64_t n3) const {
    if (!contiguous()) {
        throw std::runtime_error("ai::Tensor::reshape of a non-contiguous view " + shape_str());
    }
    if (n0 * n1 * n2 * n3 != numel()) {
        throw std::runtime_error("ai::Tensor::reshape element count mismatch " + shape_str());
    }
    Tensor r = *this;
    r.ne[0] = n0; r.ne[1] = n1; r.ne[2] = n2; r.ne[3] = n3;
    set_contiguous_strides(r);
    return r;
}

Tensor Tensor::view(int64_t n0, int64_t n1, int64_t n2, int64_t n3,
                    size_t s1, size_t s2, size_t s3, size_t offset_bytes) const {
    Tensor r = *this;
    r.data  = (char *) data + offset_bytes;
    r.ne[0] = n0; r.ne[1] = n1; r.ne[2] = n2; r.ne[3] = n3;
    r.nb[0] = type_size(type);
    r.nb[1] = s1; r.nb[2] = s2; r.nb[3] = s3;
    return r;
}

Tensor Tensor::permute(int a0, int a1, int a2, int a3) const {
    // ggml_permute semantics: source axis i moves to position axis[i]
    const int ax[4] = {a0, a1, a2, a3};
    Tensor r = *this;
    for (int i = 0; i < 4; ++i) {
        r.ne[ax[i]] = ne[i];
        r.nb[ax[i]] = nb[i];
    }
    return r;
}

std::string Tensor::shape_str() const {
    char buf[160];
    snprintf(buf, sizeof(buf), "%s[%lld,%lld,%lld,%lld]", dtype_name(type),
             (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
    return buf;
}

}  // namespace ai
