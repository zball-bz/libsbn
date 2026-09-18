#pragma once
#include <stddef.h>
#include <stdint.h>

namespace sbn::v3::coefficient_bound {
// Ten <2^48 primes need at most 480 bits. A uint64 term count times
// (2^T-1)^2, T<=256, needs at most 576 bits. This is setup-only arithmetic.
struct Wide {
    static constexpr unsigned words = 9;
    uint64_t limb[words]{};
};

constexpr Wide prime_product(const uint64_t *primes, unsigned count) noexcept {
    Wide out{};
    out.limb[0] = 1;
    for (unsigned p = 0; p < count; ++p) {
        uint64_t carry = 0;
        for (unsigned j = 0; j < Wide::words; ++j) {
            const __uint128_t value = __uint128_t(out.limb[j]) * primes[p] + carry;
            out.limb[j] = uint64_t(value);
            carry = uint64_t(value >> 64);
        }
    }
    return out;
}

inline Wide shifted(uint64_t value, unsigned bits) noexcept {
    Wide out{};
    const unsigned word = bits / 64, shift = bits % 64;
    out.limb[word] = value << shift;
    if (shift) out.limb[word + 1] = value >> (64 - shift);
    return out;
}

// Exact test: terms * (2^bits - 1)^2 < product. The expansion
// terms*2^(2bits) - terms*2^(bits+1) + terms fits in Wide without truncation.
inline bool fits(const Wide &product, unsigned bits, uint64_t terms) noexcept {
    if (bits > 256) return false;
    if (!bits || !terms) {
        for (uint64_t word : product.limb) if (word) return true;
        return false;
    }
    Wide bound = shifted(terms, 2 * bits);
    const Wide subtract = shifted(terms, bits + 1);
    uint64_t borrow = 0;
    for (unsigned j = 0; j < Wide::words; ++j) {
        const __uint128_t rhs = __uint128_t(subtract.limb[j]) + borrow;
        borrow = __uint128_t(bound.limb[j]) < rhs;
        bound.limb[j] -= uint64_t(rhs);
    }
    uint64_t carry = terms;
    for (unsigned j = 0; carry && j < Wide::words; ++j) {
        const __uint128_t value = __uint128_t(bound.limb[j]) + carry;
        bound.limb[j] = uint64_t(value);
        carry = uint64_t(value >> 64);
    }
    for (unsigned j = Wide::words; j-- > 0;) {
        if (bound.limb[j] != product.limb[j]) return bound.limb[j] < product.limb[j];
    }
    return false;
}
} // namespace sbn::v3::coefficient_bound
