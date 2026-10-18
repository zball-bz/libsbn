// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/float.hpp"
#include <bit>
#include <cstdint>
#include <limits>

namespace sbn::v3::floating {
// A private nonnegative upper bound, not an uncertain point value.
// Positive finite values are m * 2^(e-63), 2^63 <= m < 2^64.
// e is the exponent of the top bit; e-63 is evaluated only in wider
// arithmetic. This includes the minimum positive bound 2^INT64_MIN.
// All operations round outward, use fixed-size storage, and never allocate.
class ErrorMagnitude {
    using wide = __int128_t;
    using unsigned_wide = __uint128_t;
    static constexpr std::uint64_t high_bit = std::uint64_t(1) << 63;
    static constexpr std::int64_t lowest = std::numeric_limits<std::int64_t>::min();
    static constexpr std::int64_t highest = std::numeric_limits<std::int64_t>::max();

    std::uint64_t significand_ = 0;
    std::int64_t exponent2_ = 0;
    bool infinite_ = false;

    constexpr ErrorMagnitude(std::uint64_t m, std::int64_t e, bool infinite) noexcept
        : significand_(m), exponent2_(e), infinite_(infinite) {}

    // m is already normalized. If its top exponent is below the available
    // range, the represented positive value is strictly below 2^lowest:
    // rounding it UP to that minimum preserves an upper bound. Overflow
    // similarly becomes infinity. Neither case wraps an exponent or turns
    // a positive bound into exact zero.
    static constexpr ErrorMagnitude bounded(std::uint64_t m, wide e) noexcept {
        if (e < wide(lowest)) return {high_bit, lowest, false};
        if (e > wide(highest)) return {0, 0, true};
        return {m, std::int64_t(e), false};
    }

public:
    constexpr ErrorMagnitude() noexcept = default;
    static constexpr ErrorMagnitude zero() noexcept { return {}; }
    static constexpr ErrorMagnitude infinity() noexcept { return {0, 0, true}; }
    static constexpr ErrorMagnitude power_of_two(std::int64_t e) noexcept {
        return {high_bit, e, false};
    }
    // Exact normalization when the result fits; otherwise outward saturation.
    // The argument denotes units * 2^scale, including zero at any scale.
    static constexpr ErrorMagnitude from_scaled_uint64(std::uint64_t units,
                                                        std::int64_t scale = 0) noexcept {
        if (!units) return {};
        const unsigned top = 63u - unsigned(std::countl_zero(units));
        return bounded(units << (63u - top), wide(scale) + top);
    }
    static constexpr ErrorMagnitude from(error_bound error) noexcept {
        switch (error.kind) {
            case error_kind::exact: return {};
            case error_kind::finite: return power_of_two(error.exponent2);
            case error_kind::unbounded: return infinity();
        }
        return infinity();
    }

    constexpr bool is_zero() const noexcept { return !infinite_ && !significand_; }
    constexpr bool is_infinite() const noexcept { return infinite_; }
    constexpr bool is_finite() const noexcept { return !infinite_; }
    constexpr std::uint64_t significand() const noexcept { return significand_; }
    // Top-bit exponent, not the low-bit scale used by point mantissas.
    constexpr std::int64_t exponent2() const noexcept { return exponent2_; }

    constexpr ErrorMagnitude add(ErrorMagnitude other) const noexcept {
        if (infinite_ || other.infinite_) return infinity();
        if (is_zero()) return other;
        if (other.is_zero()) return *this;
        if (exponent2_ < other.exponent2_) return other.add(*this);

        const wide distance = wide(exponent2_) - other.exponent2_;
        std::uint64_t aligned;
        if (distance >= 64) {
            // The smaller positive significand contributes strictly between
            // zero and one unit at this scale, so its upward rounding is one.
            aligned = 1;
        } else {
            const unsigned shift = unsigned(distance);
            aligned = other.significand_ >> shift;
            const auto mask = (std::uint64_t(1) << shift) - 1;
            aligned += (other.significand_ & mask) != 0;
        }
        const unsigned_wide sum = unsigned_wide(significand_) + aligned;
        if (sum >> 64) {
            // Ceiling division by two preserves the already-upward sum.
            return bounded(std::uint64_t((sum + 1) >> 1), wide(exponent2_) + 1);
        }
        return bounded(std::uint64_t(sum), exponent2_);
    }

    constexpr ErrorMagnitude multiply(ErrorMagnitude other) const noexcept {
        // Error magnitudes bound finite nonnegative quantities. An exactly
        // zero factor proves a zero product even when the other bound is
        // unknown/unbounded; this defines the useful convention 0*infinity=0.
        if (is_zero() || other.is_zero()) return {};
        if (infinite_ || other.infinite_) return infinity();

        const unsigned_wide product = unsigned_wide(significand_) * other.significand_;
        const unsigned carry = unsigned(product >> 127);
        const unsigned shift = 63 + carry;
        const auto mask = (unsigned_wide(1) << shift) - 1;
        const unsigned_wide rounded = (product >> shift) + unsigned((product & mask) != 0);
        wide exponent = wide(exponent2_) + other.exponent2_ + carry;
        if (rounded >> 64) return bounded(high_bit, exponent + 1);
        return bounded(std::uint64_t(rounded), exponent);
    }

    constexpr ErrorMagnitude scale2(std::int64_t shift) const noexcept {
        if (is_zero() || infinite_) return *this;
        return bounded(significand_, wide(exponent2_) + shift);
    }

    // Publish a simple power-of-two absolute bound only at the API boundary.
    // A non-power-of-two significand needs the next exponent. At the maximum
    // exponent that next bound is unrepresentable, hence unbounded.
    constexpr error_bound publish() const noexcept {
        if (is_zero()) return {};
        if (infinite_) return error_bound::unbounded();
        if (significand_ == high_bit) return error_bound::power_of_two(exponent2_);
        if (exponent2_ == highest) return error_bound::unbounded();
        return error_bound::power_of_two(exponent2_ + 1);
    }
};
} // namespace sbn::v3::floating

