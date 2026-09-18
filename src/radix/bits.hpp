#pragma once
// Bit-level views of a dyadic mantissa (radix conversion inputs and outputs).
#include <stddef.h>
#include <stdint.h>
namespace sbn::v3::radix {
// 64 bits of m[0..count) starting at the signed bit index `bit`; bits outside the limbs are zero.
inline uint64_t bits_at(const uint64_t *m, size_t count, int64_t bit) noexcept {
    if (bit <= -64)
        return 0;
    if (bit < 0)
        return count ? m[0] << unsigned(-bit) : 0;
    const uint64_t q = uint64_t(bit) >> 6;
    const unsigned r = unsigned(bit & 63);
    if (q >= count)
        return 0;
    uint64_t v = m[q] >> r;
    if (r && q + 1 < count)
        v |= m[q + 1] << (64 - r);
    return v;
}
// out[0..n) = bits [position, position + 64 n) of the number made of the bits [low, high) of
// m[0..count) (every other bit cleared). `position` may be negative: zeros enter from below.
inline void dyadic_window(uint64_t *out, size_t n, const uint64_t *m, size_t count, int64_t position,
                          uint64_t low, uint64_t high) noexcept {
    for (size_t j = 0; j < n; ++j) {
        const int64_t at = position + int64_t(64) * int64_t(j);
        uint64_t v = bits_at(m, count, at);
        if (at < int64_t(low)) { // clear the bits below `low`
            const uint64_t drop = uint64_t(int64_t(low) - at);
            v = drop >= 64 ? 0 : v & (~uint64_t(0) << drop);
        }
        if (at + 64 > int64_t(high)) { // clear the bits at and above `high`
            const int64_t keep = int64_t(high) - at;
            v = keep <= 0 ? 0 : v & (~uint64_t(0) >> (64 - keep));
        }
        out[j] = v;
    }
}
// Number of trailing zero bits of the bits [0, high) of m; `high` when they are all zero.
inline uint64_t trailing_zero_bits(const uint64_t *m, size_t count, uint64_t high) noexcept {
    for (size_t j = 0; j < count && uint64_t(64) * j < high; ++j)
        if (m[j]) {
            const uint64_t at = uint64_t(64) * j + uint64_t(__builtin_ctzll(m[j]));
            return at < high ? at : high;
        }
    return high;
}
// (hi:lo) / d with hi < d: quotient and remainder of one schoolbook step.
inline uint64_t divide_step(uint64_t hi, uint64_t lo, uint64_t d, uint64_t &remainder) noexcept {
    uint64_t q, r;
    __asm__("divq %4" : "=a"(q), "=d"(r) : "0"(lo), "1"(hi), "r"(d) : "cc");
    remainder = r;
    return q;
}
} // namespace sbn::v3::radix
