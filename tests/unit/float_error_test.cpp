// Original libsbn tests; LGPL-3.0-or-later.
#include "float/error.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {
using sbn::v3::floating::ErrorMagnitude;
using u128 = __uint128_t;
using i128 = __int128_t;
constexpr auto low = std::numeric_limits<std::int64_t>::min();
constexpr auto high = std::numeric_limits<std::int64_t>::max();
constexpr auto high_bit = std::uint64_t(1) << 63;

unsigned bit_count(u128 n) {
    unsigned bits = 0;
    while (n) { n >>= 1; ++bits; }
    return bits;
}
// Compare two exact dyadics independently, without shifting an exponent gap
// into a buffer and without using the production floating-point comparator.
int compare_exact(u128 a, i128 ae, u128 b, i128 be) {
    if (!a || !b) return int(a != 0) - int(b != 0);
    const unsigned ab = bit_count(a), bb = bit_count(b);
    const i128 at = ae + ab, bt = be + bb;
    if (at != bt) return at > bt ? 1 : -1;
    const unsigned n = ab > bb ? ab : bb;
    for (unsigned i = 0; i < n; ++i) {
        const bool av = i < ab && ((a >> (ab - 1 - i)) & 1);
        const bool bv = i < bb && ((b >> (bb - 1 - i)) & 1);
        if (av != bv) return av ? 1 : -1;
    }
    return 0;
}
void invariant(ErrorMagnitude x) {
    if (x.is_zero() || x.is_infinite()) {
        assert(x.significand() == 0 && x.exponent2() == 0);
    } else {
        assert(x.significand() & high_bit);
    }
    const auto published = x.publish();
    if (x.is_zero()) assert(published.exact());
    else if (x.is_infinite()) assert(published.kind == sbn::error_kind::unbounded);
    else {
        assert(!published.exact());
        if (published.kind == sbn::error_kind::finite) {
            assert(compare_exact(1, published.exponent2, x.significand(),
                                 i128(x.exponent2()) - 63) >= 0);
        } else {
            assert(x.exponent2() == high && x.significand() != high_bit);
        }
    }
}
void equals(ErrorMagnitude x, u128 units, i128 scale) {
    invariant(x);
    assert(x.is_finite());
    assert(compare_exact(x.significand(), i128(x.exponent2()) - 63, units, scale) == 0);
}
void covers(ErrorMagnitude x, u128 units, i128 scale) {
    invariant(x);
    assert(x.is_finite());
    assert(compare_exact(x.significand(), i128(x.exponent2()) - 63, units, scale) >= 0);
}
void at_most(ErrorMagnitude x, u128 units, i128 scale) {
    assert(x.is_finite());
    assert(compare_exact(x.significand(), i128(x.exponent2()) - 63, units, scale) <= 0);
}
std::uint64_t random_word() {
    static std::uint64_t state = UINT64_C(0x4552524f52424e44);
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}
void elementary_and_publication() {
    static_assert(ErrorMagnitude::zero().publish().exact());
    static_assert(ErrorMagnitude::power_of_two(0).add(
                  ErrorMagnitude::power_of_two(0)).publish().exponent2 == 1);
    static_assert(ErrorMagnitude::power_of_two(-7).multiply(
                  ErrorMagnitude::power_of_two(9)).publish().exponent2 == 2);
    static_assert(ErrorMagnitude::power_of_two(low).scale2(-1).publish().exponent2 == low);
    static_assert(ErrorMagnitude::power_of_two(high).scale2(1).is_infinite());

    constexpr std::array<std::int64_t, 13> exponents{
        low, low + 1, -1000, -65, -64, -1, 0, 1, 63, 64, 1000, high - 1, high};
    for (auto e : exponents) {
        const auto x = ErrorMagnitude::power_of_two(e);
        equals(x, 1, e);
        assert(x.publish().kind == sbn::error_kind::finite);
        assert(x.publish().exponent2 == e);
        equals(ErrorMagnitude::from(x.publish()), 1, e);
        equals(x.add(ErrorMagnitude::zero()), 1, e);
        equals(ErrorMagnitude::zero().add(x), 1, e);
        equals(x.multiply(ErrorMagnitude::from_scaled_uint64(1)), 1, e);
    }
    for (std::uint64_t units : {UINT64_C(0), UINT64_C(1), UINT64_C(2), UINT64_C(3),
                                UINT64_C(7), high_bit, UINT64_MAX}) {
        for (std::int64_t scale : {-1000, -64, -1, 0, 1, 64, 1000}) {
            auto x = ErrorMagnitude::from_scaled_uint64(units, scale);
            equals(x, units, scale);
            equals(x.scale2(73).scale2(-73), units, scale);
        }
    }
    equals(ErrorMagnitude::from_scaled_uint64(0, low), 0, 0);
    equals(ErrorMagnitude::from_scaled_uint64(0, high), 0, 0);
    assert(ErrorMagnitude::from(sbn::error_bound{}).is_zero());
    assert(ErrorMagnitude::from(sbn::error_bound::unbounded()).is_infinite());
    const auto one_and_half = ErrorMagnitude::from_scaled_uint64(3, -1);
    assert(one_and_half.publish().exponent2 == 1);
    assert(ErrorMagnitude::from_scaled_uint64(UINT64_MAX).publish().exponent2 == 64);
}
void addition_exact_witnesses() {
    for (unsigned i = 0; i < 3000; ++i) {
        const std::uint64_t a = random_word() | high_bit;
        const std::uint64_t b = random_word() | high_bit;
        const unsigned gap = unsigned(random_word() % 63);
        const auto ae = std::int64_t(random_word() % 201) - 100;
        const auto be = ae - gap;
        const auto av = ErrorMagnitude::from_scaled_uint64(a, ae);
        const auto bv = ErrorMagnitude::from_scaled_uint64(b, be);
        const auto sum = av.add(bv);
        const u128 exact = (u128(a) << gap) + b;
        covers(sum, exact, be);
        // The exact reference fits in 127 bits here. Even after alignment and
        // normalization, the excess must fit one unit of the result mantissa.
        const i128 shift = i128(sum.exponent2()) - 63 - be;
        assert(shift >= 0 && shift <= 63);
        at_most(sum, exact + (u128(1) << unsigned(shift)), be);
        const auto reverse = bv.add(av);
        assert(sum.significand() == reverse.significand());
        assert(sum.exponent2() == reverse.exponent2());
    }
    // The 64-bit alignment boundary, including a carry to 2^128.
    const auto a = ErrorMagnitude::from_scaled_uint64(UINT64_MAX, 64);
    const auto b = ErrorMagnitude::from_scaled_uint64(UINT64_MAX, 0);
    const auto sum = a.add(b);
    covers(sum, ~u128(0), 0);
    equals(sum, 1, 128);

    auto accumulated = ErrorMagnitude::zero();
    const auto one = ErrorMagnitude::power_of_two(0);
    for (unsigned n = 1; n <= 10000; ++n) {
        accumulated = accumulated.add(one);
        equals(accumulated, n, 0);
    }
    // Repeatedly adding tiny positive errors must not double the bound every
    // time. A normalized significand retains useful density in the same binade.
    accumulated = one;
    const auto tiny = ErrorMagnitude::power_of_two(-1000);
    for (unsigned n = 0; n < 10000; ++n) accumulated = accumulated.add(tiny);
    invariant(accumulated);
    assert(accumulated.is_finite() && !accumulated.is_zero());
    assert(accumulated.publish().exponent2 <= 1);
    assert(compare_exact(accumulated.significand(), i128(accumulated.exponent2()) - 63,
                         1, 0) > 0);
}
void multiplication_exact_witnesses() {
    for (unsigned i = 0; i < 3000; ++i) {
        const std::uint64_t a = random_word() | high_bit;
        const std::uint64_t b = random_word() | high_bit;
        const auto ae = std::int64_t(random_word() % 201) - 100;
        const auto be = std::int64_t(random_word() % 201) - 100;
        const auto av = ErrorMagnitude::from_scaled_uint64(a, ae);
        const auto bv = ErrorMagnitude::from_scaled_uint64(b, be);
        const auto product = av.multiply(bv);
        const u128 exact = u128(a) * b;
        covers(product, exact, ae + be);
        const i128 shift = i128(product.exponent2()) - 63 - (ae + be);
        assert(shift >= 63 && shift <= 64);
        // For a,b <= UINT64_MAX, exact + 2^64 is still below 2^128.
        at_most(product, exact + (u128(1) << unsigned(shift)), ae + be);
        const auto reverse = bv.multiply(av);
        assert(product.significand() == reverse.significand());
        assert(product.exponent2() == reverse.exponent2());
    }
    const auto almost_two = ErrorMagnitude::from_scaled_uint64(UINT64_MAX, -63);
    covers(almost_two.multiply(almost_two), u128(UINT64_MAX) * UINT64_MAX, -126);
    equals(ErrorMagnitude::from_scaled_uint64(3, -1).multiply(
           ErrorMagnitude::from_scaled_uint64(3, -1)), 9, -2);
}
void range_and_special_states() {
    const auto zero = ErrorMagnitude::zero();
    const auto inf = ErrorMagnitude::infinity();
    const auto smallest = ErrorMagnitude::power_of_two(low);
    const auto largest_power = ErrorMagnitude::power_of_two(high);
    const auto half = ErrorMagnitude::power_of_two(-1);
    const auto two = ErrorMagnitude::power_of_two(1);
    assert(inf.add(zero).is_infinite() && zero.add(inf).is_infinite());
    assert(inf.add(smallest).is_infinite());
    assert(inf.multiply(two).is_infinite());
    assert(inf.multiply(zero).is_zero() && zero.multiply(inf).is_zero());
    assert(inf.scale2(low).is_infinite());
    assert(zero.scale2(low).is_zero() && zero.scale2(high).is_zero());

    equals(smallest.scale2(-1), 1, low);
    equals(smallest.scale2(low), 1, low);
    equals(smallest.multiply(half), 1, low);
    equals(smallest.multiply(smallest), 1, low);
    equals(smallest.multiply(largest_power), 1, -1);
    equals(largest_power.scale2(low), 1, -1);
    equals(smallest.scale2(high), 1, -1);

    assert(largest_power.add(largest_power).is_infinite());
    assert(largest_power.multiply(two).is_infinite());
    assert(largest_power.multiply(largest_power).is_infinite());
    assert(largest_power.scale2(1).is_infinite());
    assert(ErrorMagnitude::from_scaled_uint64(2, high).is_infinite());

    const auto almost_limit = ErrorMagnitude::from_scaled_uint64(UINT64_MAX, high - 63);
    invariant(almost_limit);
    assert(almost_limit.is_finite());
    assert(almost_limit.publish().kind == sbn::error_kind::unbounded);
    assert(almost_limit.scale2(-1).publish().kind == sbn::error_kind::finite);
    assert(almost_limit.scale2(-1).publish().exponent2 == high);
    assert(almost_limit.add(smallest).is_infinite());

    const auto min_fraction = smallest.multiply(ErrorMagnitude::from_scaled_uint64(3, -1));
    covers(min_fraction, 3, i128(low) - 1);
    assert(!min_fraction.is_zero());
    assert(min_fraction.publish().kind == sbn::error_kind::finite);
    assert(min_fraction.publish().exponent2 == low + 1);
}
} // namespace

int main() {
    elementary_and_publication();
    addition_exact_witnesses();
    multiplication_exact_witnesses();
    range_and_special_states();
    std::puts("float error magnitude: exact dyadic witnesses, outward arithmetic, "
              "dense accumulation and exponent saturation PASS");
}
