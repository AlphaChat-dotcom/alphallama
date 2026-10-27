#pragma once
// Minimal scalar_type.hpp for Marlin standalone (no PyTorch, no C++20)
// Only provides what the Marlin kernel template needs: ScalarTypeId, kU4, kU8

#include <cstdint>

namespace sglang {

class ScalarType {
public:
    using Id = uint64_t;

    uint8_t exponent_;
    uint8_t mantissa_;
    bool signed_;
    uint8_t bias_;

    constexpr ScalarType(uint8_t e, uint8_t m, bool s, uint8_t bias)
        : exponent_(e), mantissa_(m), signed_(s), bias_(bias) {}

    static constexpr ScalarType uint(uint8_t bits, uint8_t pack = 0) {
        return ScalarType(0, bits, false, 0);
    }

    static constexpr ScalarType float_IEEE754(uint8_t e, uint8_t m) {
        return ScalarType(e, m, true, (1 << (e - 1)) - 1);
    }

    constexpr Id id() const {
        return (uint64_t)exponent_ | ((uint64_t)mantissa_ << 8) |
               ((uint64_t)signed_ << 16) | ((uint64_t)bias_ << 24);
    }

    static constexpr ScalarType from_id(Id id) {
        return ScalarType(id & 0xFF, (id >> 8) & 0xFF, (id >> 16) & 1, (id >> 24) & 0xFF);
    }

    constexpr int size_bits() const {
        return exponent_ + mantissa_ + (signed_ ? 1 : 0);
    }

    constexpr bool is_integer() const { return exponent_ == 0; }
    constexpr bool is_signed() const { return signed_; }
    constexpr bool is_ieee_754() const { return exponent_ > 0; }
    constexpr bool has_nans() const { return is_ieee_754(); }
    constexpr bool has_infs() const { return is_ieee_754(); }
    constexpr bool has_bias() const { return bias_ > 0; }
    constexpr int mantissa() const { return mantissa_; }
    constexpr int exponent() const { return exponent_; }

    constexpr bool operator==(const ScalarType& o) const { return id() == o.id(); }
    constexpr bool operator!=(const ScalarType& o) const { return id() != o.id(); }
};

using ScalarTypeId = ScalarType::Id;

// Integer types
static inline constexpr auto kU4 = ScalarType::uint(4);
static inline constexpr auto kU4B8 = ScalarType(0, 4, false, 8);  // uint4 with bias=8
static inline constexpr auto kU8 = ScalarType::uint(8);
static inline constexpr auto kU8B128 = ScalarType(0, 8, false, 128);  // uint8 with bias=128
static inline constexpr auto kUint4 = kU4;
static inline constexpr auto kUint4b8 = kU4B8;

// FP types used by Marlin for type checks
static inline constexpr auto kFE2M1f = ScalarType::float_IEEE754(2, 1);
static inline constexpr auto kFE3M2f = ScalarType::float_IEEE754(3, 2);
static inline constexpr auto kFE4M3fn = ScalarType::float_IEEE754(4, 3);
static inline constexpr auto kFE8M0fnu = ScalarType(8, 0, false, 127);

}  // namespace sglang
