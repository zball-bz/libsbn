#pragma once
#include <stdint.h>
namespace sbn::v3::bbp {
constexpr unsigned radix_bits = 48, max_digits = 8;
constexpr uint64_t mask = (uint64_t(1) << radix_bits) - 1;
// pi = sum_j sum_k (-1)^(k+negative_j) 2^(shift_j-10k)/(a_j*k+b_j).
struct Term {
    unsigned a, b;
    int shift;
    bool negative;
};
inline constexpr Term formula[7] = {{4, 1, -1, true},  {4, 3, -6, true},  {10, 1, 2, false}, {10, 3, 0, true},
                                    {10, 5, -4, true}, {10, 7, -4, true}, {10, 9, -6, false}};
struct Fixed {
    uint64_t d[max_digits]{}; // little-endian radix 2^48, modulo 2^window_bits
    void add(const Fixed &x, unsigned n, bool negative = false) {
        int64_t carry = 0;
        for (unsigned j = 0; j < n; ++j) {
            carry += int64_t(d[j]) + (negative ? -int64_t(x.d[j]) : int64_t(x.d[j]));
            d[j] = uint64_t(carry) & mask;
            carry >>= radix_bits; // C++20 arithmetic shift, including negative carries
        }
    }
    void adjust(uint64_t radius, unsigned n, bool negative) {
        Fixed x{};
        for (unsigned j = 0; radius; ++j) {
            x.d[j] = radius & mask;
            radius >>= radix_bits;
        }
        add(x, n, negative);
    }
    uint64_t extract(unsigned low) const {
        const unsigned word = low / radix_bits, shift = low % radix_bits;
        __uint128_t v = 0;
        for (unsigned j = 0; j < 3 && word + j < max_digits; ++j)
            v |= __uint128_t(d[word + j]) << (radix_bits * j);
        return uint64_t(v >> shift);
    }
};
inline uint64_t main_terms(uint64_t offset, Term t) {
    const int64_t e = int64_t(offset) + t.shift;
    return e < 0 ? 0 : uint64_t(e) / 10 + 1;
}
inline uint64_t pow2(uint64_t e, uint64_t d) {
    uint64_t r = 1 % d, a = 2 % d;
    while (e) {
        if (e & 1)
            r = __uint128_t(r) * a % d;
        e >>= 1;
        if (e)
            a = __uint128_t(a) * a % d;
    }
    return r;
}
inline Fixed scalar_term(uint64_t offset, Term t, uint64_t k, unsigned n) {
    const uint64_t denominator = t.a * k + t.b;
    const int64_t e = int64_t(offset) + t.shift - int64_t(10 * k);
    Fixed out{};
    if (e >= 0) {
        uint64_t r = pow2(uint64_t(e), denominator);
        for (unsigned j = n; j--;) {
            const __uint128_t v = __uint128_t(r) << radix_bits;
            out.d[j] = v / denominator;
            r = v % denominator;
        }
    } else if (e + int64_t(radix_bits * n) >= 0) {
        const unsigned bit = unsigned(e + int64_t(radix_bits * n));
        out.d[bit / radix_bits] = uint64_t(1) << (bit % radix_bits);
        uint64_t r = 0;
        for (unsigned j = n; j--;) {
            const __uint128_t v = (__uint128_t(r) << radix_bits) + out.d[j];
            out.d[j] = v / denominator;
            r = v % denominator;
        }
    }
    return out;
}
// Native kernel accepts main terms only, count <= 16384. Exact fixed-point
// floors, identical to scalar_term; SIMD rounding is corrected with integers.
void native_sum(Fixed &, uint64_t offset, Term, uint64_t begin, uint64_t end, unsigned digits);
} // namespace sbn::v3::bbp
