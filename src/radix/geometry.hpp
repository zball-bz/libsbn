#pragma once
// Geometry and precision ledger of the radix conversion trees
// (docs/radix-conversion-design-2026-09-18.md). Pure functions of the base
// and the fragment count; query, bind and the tests all derive from them.
#include "radix/leaf.hpp"
#include "common/checked.hpp"
#include <cmath>
namespace sbn::v3::radix {
// Bits kept beyond ceil(log2 b^d) by a node that delivers d digits. The
// Bouvier-Zimmermann invariant loses 1/(4g) of the last digit per right
// descent (g >= depth); 16 bits cover 2^34 digits with room for the rounding
// of the size bounds below. Limb granularity usually adds more.
inline constexpr unsigned guard_bits = 16;
// Nodes of at most this many fragments are converted by one leaf group call.
inline constexpr uint64_t group_fragments = 8;
inline constexpr uint64_t max_fragments = uint64_t(1) << 34; // 2^40 digits
struct BaseInfo {
    unsigned base = 0, twos = 0, odd = 1; // base = 2^twos * odd
    long double log2_base = 0, log2_odd = 0;
};
inline bool base_info(unsigned base, BaseInfo &out) noexcept {
    out = {};
    if (base < 2 || base > 64)
        return false;
    out.base = base;
    out.odd = base;
    while (!(out.odd & 1)) {
        out.odd >>= 1;
        ++out.twos;
    }
    out.log2_base = log2l((long double)base);
    out.log2_odd = log2l((long double)out.odd);
    return true;
}
// An upper bound of ceil(log2 value^count), at most 2 above it.
inline uint64_t power_bits(long double log2_value, uint64_t count) noexcept {
    return uint64_t(floorl(log2_value * (long double)count)) + 2;
}
inline size_t limbs_for_bits(uint64_t bits) noexcept {
    return size_t((bits + 63) / 64);
}
// Capacity of the rail entry odd^(64 * 2^k) in limbs (its top limb may be zero).
inline size_t rail_limbs(const BaseInfo &b, unsigned k) noexcept {
    return std::max<size_t>(1, limbs_for_bits(power_bits(b.log2_odd, uint64_t(fragment_digits) << k)));
}
// Fraction limbs of a node that delivers 64*fragments digits and one overlap word.
inline size_t node_limbs(const BaseInfo &b, uint64_t fragments) noexcept {
    return limbs_for_bits(power_bits(b.log2_base, fragments * fragment_digits + word_digits) + guard_bits);
}
// Limbs of an integer below base^(64*fragments).
inline size_t integer_limbs(const BaseInfo &b, uint64_t fragments) noexcept {
    return limbs_for_bits(power_bits(b.log2_base, fragments * fragment_digits));
}
// The perfect child of a ragged-right split: the largest power of two below n (n >= 2).
inline unsigned split_level(uint64_t n) noexcept {
    return 63 - unsigned(__builtin_clzll(n - 1));
}
// The right child's fraction is bits [window, window + 64*right_limbs) of the
// exact product (node fraction) * odd^(64*2^k): the power of two of the base
// moves the binary point instead of the operand.
inline uint64_t window_bit(const BaseInfo &b, size_t limbs, size_t right_limbs, unsigned k) noexcept {
    const uint64_t shift = (uint64_t(b.twos) * fragment_digits) << k;
    const uint64_t high = uint64_t(64) * (limbs - right_limbs);
    require(high >= shift, SBN3_FATAL_MATH, "radix window below the product");
    return high - shift;
}
// out[0..n) = bits [bit, bit + 64 n) of value[0..count), zero beyond the top; truncation toward zero.
inline void take_window(uint64_t *out, size_t n, const uint64_t *value, size_t count, uint64_t bit) noexcept {
    const size_t q = size_t(bit >> 6);
    const unsigned r = unsigned(bit & 63);
    for (size_t i = 0; i < n; ++i) {
        const uint64_t lo = q + i < count ? value[q + i] : 0;
        const uint64_t hi = q + i + 1 < count ? value[q + i + 1] : 0;
        out[i] = r ? (lo >> r) | (hi << (64 - r)) : lo;
    }
}
} // namespace sbn::v3::radix
