#pragma once
#include "verify/bbp.hpp"
#include <stddef.h>

namespace sbn::v3::bbp {
constexpr unsigned modulus_words = 4, max_modulus_bits = 64 * modulus_words;
// Internal scalar ABI between formula lowering and arithmetic backends.
// A job contributes sign * numerator * 2^exponent / modulus, modulo one.
// Positive (possibly even) modulus, arbitrary u64 numerator and lane exponent.
struct RationalTerm {
    uint64_t modulus[modulus_words];
    uint64_t numerator;
    int64_t exponent;
    bool negative;
};

struct WideModulus {
    uint64_t d[modulus_words]{};
    unsigned bits() const {
        for (unsigned j = modulus_words; j--;)
            if (d[j])
                return 64 * j + 64 - __builtin_clzll(d[j]);
        return 0;
    }
    bool multiply(uint64_t x) {
        __uint128_t carry = 0;
        for (auto &v : d) {
            carry += __uint128_t(v) * x;
            v = uint64_t(carry);
            carry >>= 64;
        }
        return !carry;
    }
    uint64_t extract(unsigned bit) const {
        const unsigned word = bit / 64, shift = bit % 64;
        uint64_t v = word < modulus_words ? d[word] >> shift : 0;
        if (shift && word + 1 < modulus_words)
            v |= d[word + 1] << (64 - shift);
        return v;
    }
    unsigned strip_twos() {
        unsigned low = 0;
        while (!d[low])
            ++low; // Positive modulus is an entry precondition.
        const unsigned shift = 64 * low + __builtin_ctzll(d[low]);
        for (unsigned j = 0; j < modulus_words; ++j)
            d[j] = extract(shift + 64 * j);
        return shift;
    }
    // r <- 2r + bit (mod m), r < m; return the binary quotient digit.
    bool step(unsigned bit, const WideModulus &m) {
        uint64_t carry = bit;
        for (auto &v : d) {
            const uint64_t next = v >> 63;
            v = (v << 1) | carry;
            carry = next;
        }
        bool take = carry != 0;
        if (!take) {
            take = true;
            for (unsigned j = modulus_words; j--;)
                if (d[j] != m.d[j]) {
                    take = d[j] > m.d[j];
                    break;
                }
        }
        if (take) {
            uint64_t borrow = 0;
            for (unsigned j = 0; j < modulus_words; ++j) {
                const __uint128_t v = __uint128_t(m.d[j]) + borrow;
                borrow = __uint128_t(d[j]) < v;
                d[j] -= uint64_t(v);
            }
        }
        return take;
    }
};

// Short negative-exponent tail. Exact integer long division, independent of
// Montgomery and FMA. The quotient is kept modulo 2^(48*digits).
inline Fixed scaled_tail(WideModulus m, uint64_t numerator, int64_t exponent, unsigned digits) {
    Fixed out{};
    const unsigned width = digits * radix_bits;
    int64_t shift = exponent + width;
    if (shift < 0) {
        numerator = shift <= -64 ? 0 : numerator >> unsigned(-shift);
        shift = 0;
    }
    if (!numerator)
        return out;
    const unsigned length = 64 - __builtin_clzll(numerator) + unsigned(shift);
    WideModulus remainder{};
    for (unsigned bit = length; bit--;) {
        const unsigned input = bit >= unsigned(shift) ? (numerator >> (bit - shift)) & 1 : 0;
        if (remainder.step(input, m) && bit < width)
            out.d[bit / radix_bits] |= uint64_t(1) << (bit % radix_bits);
    }
    return out;
}

// Native u52 CIOS, L=1..5, exact radix-2^48 fraction output. Width buckets are
// filled across input streams and differing exponents; only bucket tails mask.
// count can be any size. Caller bounds batches for scheduling, not arithmetic.
void native_rational_sum(Fixed &, const RationalTerm *, size_t count, unsigned digits);

// Native factored-stream producer. Query guarantees odd q=scale*(a*k+b)^power,
// 1 <= q < 2^104, numerator < 2^52, and nonnegative exponent-stride*k.
// a>=0, b/scale>0, 1<=power<=8, monotone q; at most 16384 terms per call.
// The root a*k+b fits u64. Split at q=2^52; both widths avoid per-term job records.
struct WordStream {
    uint64_t a, b, scale, numerator;
    int64_t exponent;
    unsigned stride, power;
    bool alternating, negative;
};
void native_word_sum(Fixed &, const WordStream &, uint64_t begin, uint64_t end, unsigned digits);
} // namespace sbn::v3::bbp
