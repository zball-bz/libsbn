// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/float.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();

namespace {
using u128 = __uint128_t;
using i128 = __int128_t;
using sbn::rounding;
constexpr std::array modes{
    rounding::nearest_even, rounding::toward_zero, rounding::upward,
    rounding::downward, rounding::away_zero};
constexpr std::array<unsigned, 14> precisions{
    1, 2, 3, 31, 32, 53, 63, 64, 65, 96, 127, 128, 129, 257};
std::size_t checked_operations = 0;
alignas(64) std::array<std::byte, 4096> scratch{};

// These witnesses are plain limb arrays; construction/comparison does not
// call production integer arithmetic or the production float comparator.
struct point {
    std::array<std::uint64_t, 40> words{};
    std::int64_t exponent = 0;
    bool negative = false;
    sbn::float_view view() const noexcept {
        return sbn::float_view::from_magnitude(words, exponent, negative);
    }
    void bit(unsigned n) noexcept {
        assert(n / 64 < words.size());
        words[n / 64] |= std::uint64_t(1) << (n % 64);
    }
    void increment() noexcept {
        for (auto& word : words) {
            if (++word) return;
        }
        assert(false);
    }
};
point small(u128 magnitude, std::int64_t exponent = 0, bool negative = false) {
    point r;
    r.words[0] = std::uint64_t(magnitude);
    r.words[1] = std::uint64_t(magnitude >> 64);
    r.exponent = exponent;
    r.negative = negative;
    return r;
}
unsigned word_bits(std::span<const std::uint64_t> w) {
    std::size_t n = w.size();
    while (n && w[n - 1] == 0) --n;
    if (!n) return 0;
    unsigned high = 0;
    for (auto v = w[n - 1]; v; v >>= 1) ++high;
    return unsigned((n - 1) * 64) + high;
}
unsigned bits128(u128 v) {
    unsigned n = 0;
    while (v) { v >>= 1; ++n; }
    return n;
}
unsigned ceil_log2(u128 v) {
    assert(v);
    return bits128(v - 1);
}
bool bit_from_top(std::span<const std::uint64_t> w, unsigned n, unsigned i) {
    if (i >= n) return false;
    const unsigned at = n - 1 - i;
    return (w[at / 64] >> (at % 64)) & 1;
}
void equal_point(sbn::float_view got, const point& expected) {
    assert(got.finite());
    const auto ew = std::span<const std::uint64_t>(expected.words);
    const auto gw = got.magnitude().limbs();
    const unsigned eb = word_bits(ew), gb = word_bits(gw);
    if (!eb) {
        assert(got.zero());
        assert(got.negative() == expected.negative);
        return;
    }
    assert(gb);
    assert(got.negative() == expected.negative);
    assert(i128(got.exponent2()) + gb == i128(expected.exponent) + eb);
    const unsigned n = eb > gb ? eb : gb;
    for (unsigned i = 0; i < n; ++i)
        assert(bit_from_top(gw, gb, i) == bit_from_top(ew, eb, i));
}
void finite_precision(sbn::float_ref out) {
    const auto v = out.view();
    if (v.finite()) assert(word_bits(v.magnitude().limbs()) <= out.precision().count);
}
sbn::float_report rounded(sbn::float_ref out, sbn::float_view x,
                         rounding mode = rounding::nearest_even) {
    allocation_watch_start();
    const auto report = sbn::round_into(out, x, mode);
    assert(allocation_watch_stop() == 0);
    ++checked_operations;
    finite_precision(out);
    return report;
}
sbn::float_report added(sbn::float_ref out, sbn::float_view a, sbn::float_view b,
                       rounding mode = rounding::nearest_even, bool subtract = false) {
    allocation_watch_start();
    const auto need = sbn::add_requirements(out, a, b);
    assert(need.supported());
    assert(need.output_limbs == (out.precision().count + 63) / 64);
    assert(need.scratch_bytes <= scratch.size());
    assert(need.scratch_alignment <= 64);
    const sbn::execution ex{{scratch.data(), need.scratch_bytes}, nullptr, nullptr};
    const auto report = subtract ? sbn::subtract_into(out, a, b, ex, mode)
                                 : sbn::add_into(out, a, b, ex, mode);
    assert(allocation_watch_stop() == 0);
    ++checked_operations;
    finite_precision(out);
    return report;
}
void error_at_scale(const sbn::float_report& report, u128 difference,
                    std::int64_t exponent) {
    if (!difference) {
        assert(report.error.exact());
        assert(!sbn::has_flag(report.flags, sbn::float_flag::rounded));
    } else {
        assert(report.error.kind == sbn::error_kind::finite);
        assert(i128(report.error.exponent2) >= i128(exponent) + ceil_log2(difference));
        assert(sbn::has_flag(report.flags, sbn::float_flag::rounded));
    }
    assert(!sbn::has_flag(report.flags, sbn::float_flag::invalid));
    assert(!sbn::has_flag(report.flags, sbn::float_flag::overflow));
    assert(!sbn::has_flag(report.flags, sbn::float_flag::underflow));
}

// Independent small exact oracle. Shifts are performed on unsigned values;
// all signed test values and sums deliberately fit well below 127 bits.
u128 magnitude(i128 x) {
    const u128 u = u128(x);
    return x < 0 ? u128(0) - u : u;
}
i128 signed_value(u128 m, bool negative) {
    assert(bits128(m) < 127);
    return negative ? -i128(m) : i128(m);
}
point reference_round(i128 value, std::int64_t scale, unsigned p,
                      rounding mode, bool zero_negative = false) {
    if (!value) return small(0, 0, zero_negative);
    const bool negative = value < 0;
    const u128 m = magnitude(value);
    const unsigned n = bits128(m);
    if (n <= p) return small(m, scale, negative);
    const unsigned cut = n - p;
    const u128 divisor = u128(1) << cut;
    u128 q = m / divisor;
    const u128 r = m % divisor;
    bool up = false;
    switch (mode) {
        case rounding::nearest_even:
            up = r > divisor / 2 || (r == divisor / 2 && (q & 1));
            break;
        case rounding::toward_zero: break;
        case rounding::upward: up = r && !negative; break;
        case rounding::downward: up = r && negative; break;
        case rounding::away_zero: up = r != 0; break;
    }
    q += unsigned(up);
    // The reference can retain a carry bit; equal_point compares exact values,
    // while finite_precision separately enforces the public precision contract.
    return small(q, scale + cut, negative);
}
i128 value_at_scale(sbn::float_view v, std::int64_t scale) {
    assert(v.finite());
    u128 m = 0;
    const auto words = v.magnitude().limbs();
    for (std::size_t j = 0; j < words.size(); ++j) {
        for (unsigned k = 0; k < 64; ++k) {
            if (!((words[j] >> k) & 1)) continue;
            const i128 shift = i128(v.exponent2()) - scale + 64 * j + k;
            assert(shift >= 0 && shift < 126);
            m |= u128(1) << unsigned(shift);
        }
    }
    return signed_value(m, v.negative());
}
void check_reference(sbn::float_ref out, const sbn::float_report& report,
                     i128 exact, std::int64_t scale, unsigned p,
                     rounding mode, bool zero_negative = false) {
    equal_point(out.view(), reference_round(exact, scale, p, mode, zero_negative));
    error_at_scale(report, magnitude(value_at_scale(out.view(), scale) - exact), scale);
}
std::uint64_t random_word() {
    static std::uint64_t state = UINT64_C(0x53424e33464c4f41);
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

void small_exact_oracle() {
    for (unsigned trial = 0; trial < 400; ++trial) {
        const unsigned width = 1 + unsigned(random_word() % 120);
        u128 m = (u128(random_word()) << 64) | random_word();
        m &= (u128(1) << width) - 1;
        const bool negative = (random_word() & 1) != 0;
        const std::int64_t scale = std::int64_t(random_word() % 61) - 30;
        const auto input = small(m, scale, negative);
        const unsigned p = precisions[trial % precisions.size()];
        for (auto mode : modes) {
            std::array<std::uint64_t, 8> storage{};
            sbn::float_buffer out(storage, sbn::bits{p});
            const auto report = rounded(out.ref(), input.view(), mode);
            check_reference(out.ref(), report, signed_value(m, negative), scale,
                            p, mode, negative && !m);
        }
    }
    for (unsigned trial = 0; trial < 500; ++trial) {
        const u128 am = random_word() & ((UINT64_C(1) << 54) - 1);
        const u128 bm = random_word() & ((UINT64_C(1) << 54) - 1);
        const auto ae = std::int64_t(random_word() % 61) - 30;
        const auto be = std::int64_t(random_word() % 61) - 30;
        const bool an = (random_word() & 1) != 0;
        const bool bn = (random_word() & 1) != 0;
        const auto a = small(am, ae, an), b = small(bm, be, bn);
        const auto a_before = a.words, b_before = b.words;
        const auto base = ae < be ? ae : be;
        const i128 av = signed_value(am << unsigned(ae - base), an);
        const i128 bv = signed_value(bm << unsigned(be - base), bn);
        const unsigned p = precisions[trial % precisions.size()];
        const auto comparison = av < bv ? std::partial_ordering::less
                              : av > bv ? std::partial_ordering::greater
                                        : std::partial_ordering::equivalent;
        assert(sbn::compare(a.view(), b.view()) == comparison);
        for (auto mode : modes) for (bool subtract : {false, true}) {
            std::array<std::uint64_t, 8> storage{};
            sbn::float_buffer out(storage, sbn::bits{p});
            const i128 exact = subtract ? av - bv : av + bv;
            const auto report = added(out.ref(), a.view(), b.view(), mode, subtract);
            check_reference(out.ref(), report, exact, base, p, mode,
                            !exact && mode == rounding::downward);
            assert(a.words == a_before && b.words == b_before);
        }
    }
}

void multi_limb_rounding_witnesses() {
    for (unsigned p : precisions) for (bool negative : {false, true})
        for (unsigned parity : {0u, 1u}) for (unsigned residue : {0u, 3u, 4u, 5u})
            for (auto mode : modes) {
                // Exact value q*8+residue, q has p bits. This supplies below,
                // at, and above half-way witnesses across every limb boundary.
                point q;
                q.bit(p - 1);
                if (parity && p > 1) q.bit(0);
                point input;
                input.bit(p + 2);
                if (parity && p > 1) input.bit(3);
                input.words[0] |= residue;
                input.exponent = -17;
                input.negative = negative;
                const bool odd = (q.words[0] & 1) != 0;
                const bool increment = residue &&
                    (mode == rounding::away_zero ||
                     (mode == rounding::upward && !negative) ||
                     (mode == rounding::downward && negative) ||
                     (mode == rounding::nearest_even &&
                      (residue > 4 || (residue == 4 && odd))));
                if (increment) q.increment();
                q.exponent = -14;
                q.negative = negative;
                std::array<std::uint64_t, 8> storage{};
                sbn::float_buffer out(storage, sbn::bits{p});
                const auto report = rounded(out.ref(), input.view(), mode);
                equal_point(out.view(), q);
                error_at_scale(report, increment ? 8 - residue : residue, -17);
                // Exact-base alias, including non-word-aligned right shifts.
                auto alias = input;
                sbn::float_buffer in_place(alias.words, sbn::bits{p});
                const auto alias_report = rounded(in_place.ref(), alias.view(), mode);
                equal_point(in_place.view(), q);
                error_at_scale(alias_report, increment ? 8 - residue : residue, -17);
            }
    for (unsigned p : precisions) for (bool negative : {false, true}) {
        point input, expected;
        for (unsigned k = 0; k < p; ++k) input.bit(k + 3);
        input.bit(2); // all-one significand at exactly half-way: carry to 2^p.
        input.exponent = 7;
        input.negative = negative;
        expected.bit(p);
        expected.exponent = 10;
        expected.negative = negative;
        std::array<std::uint64_t, 8> storage{};
        sbn::float_buffer out(storage, sbn::bits{p});
        const auto report = rounded(out.ref(), input.view());
        equal_point(out.view(), expected);
        error_at_scale(report, 4, 7);
    }
}

void cancellation_and_aliases() {
    for (auto scale : {std::int64_t(0), -INT64_C(1000000000000), INT64_C(1000000000000)})
        for (auto mode : modes) {
            point a, b;
            a.bit(2048); a.bit(0);
            b.bit(2048);
            a.exponent = b.exponent = scale;
            const auto before_a = a.words, before_b = b.words;
            for (unsigned p : {1u, 65u, 129u, 257u}) {
                std::array<std::uint64_t, 8> storage{};
                sbn::float_buffer out(storage, sbn::bits{p});
                auto report = added(out.ref(), a.view(), b.view(), mode, true);
                equal_point(out.view(), small(1, scale));
                assert(report.error.exact());
                assert(a.words == before_a && b.words == before_b);
                auto alias_a = a;
                sbn::float_buffer left(alias_a.words, sbn::bits{p});
                report = added(left.ref(), alias_a.view(), b.view(), mode, true);
                equal_point(left.view(), small(1, scale));
                assert(report.error.exact());
                auto alias_b = b;
                sbn::float_buffer right(alias_b.words, sbn::bits{p});
                report = added(right.ref(), a.view(), alias_b.view(), mode, true);
                equal_point(right.view(), small(1, scale));
                assert(report.error.exact());
            }
            std::array<std::uint64_t, 8> storage{};
            sbn::float_buffer out(storage, sbn::bits{65});
            const auto report = added(out.ref(), a.view(), a.view(), mode, true);
            equal_point(out.view(), small(0, 0, mode == rounding::downward));
            assert(report.error.exact());
        }
    const auto a = small(17, -4, true), b = small(5, -6);
    for (auto mode : modes) for (bool subtract : {false, true}) {
        for (bool alias_left : {false, true}) {
            auto alias = alias_left ? a : b;
            sbn::float_buffer out(alias.words, sbn::bits{3});
            const auto report = alias_left
                ? added(out.ref(), alias.view(), b.view(), mode, subtract)
                : added(out.ref(), a.view(), alias.view(), mode, subtract);
            check_reference(out.ref(), report, subtract ? -73 : -63, -6, 3, mode);
        }
    }
}

void virtual_exponent_gaps() {
    constexpr std::int64_t gap = INT64_C(1000000000000);
    for (auto scale : {-gap, std::int64_t(0), gap})
        for (unsigned p : {1u, 65u, 129u, 257u})
            for (bool negative : {false, true}) for (bool subtract_small : {false, true})
                for (auto mode : modes) {
                    const auto a = small(1, scale, negative);
                    const auto b = small(1, scale - gap, negative != subtract_small);
                    point expected = a;
                    bool neighbor = false;
                    std::int64_t neighbor_exponent = 0;
                    if (!subtract_small) {
                        neighbor = mode == rounding::away_zero ||
                            (mode == rounding::upward && !negative) ||
                            (mode == rounding::downward && negative);
                        if (neighbor) {
                            expected = {};
                            expected.bit(p - 1);
                            expected.increment();
                            expected.exponent = scale - p + 1;
                            neighbor_exponent = expected.exponent;
                        }
                    } else {
                        neighbor = mode == rounding::toward_zero ||
                            (mode == rounding::upward && negative) ||
                            (mode == rounding::downward && !negative);
                        if (neighbor) {
                            expected = {};
                            for (unsigned k = 0; k < p; ++k) expected.bit(k);
                            expected.exponent = scale - p;
                            neighbor_exponent = expected.exponent;
                        }
                    }
                    expected.negative = negative;
                    for (bool swapped : {false, true}) {
                        std::array<std::uint64_t, 8> storage{};
                        sbn::float_buffer out(storage, sbn::bits{p});
                        const auto av = swapped ? b.view() : a.view();
                        const auto bv = swapped ? a.view() : b.view();
                        const auto need = sbn::add_requirements(out.ref(), av, bv);
                        assert(need.supported() && need.scratch_bytes < 4096);
                        const auto report = added(out.ref(), av, bv, mode);
                        equal_point(out.view(), expected);
                        assert(report.error.kind == sbn::error_kind::finite);
                        const auto minimum = neighbor ? neighbor_exponent : scale - gap;
                        assert(report.error.exponent2 >= minimum);
                        assert(sbn::has_flag(report.flags, sbn::float_flag::rounded));
                        assert(!sbn::has_flag(report.flags, sbn::float_flag::overflow));
                    }
                }
    // A tiny far-away term changes which side of an exact rounding midpoint
    // the sum lies on. Losing that term before collecting sticky bits is wrong.
    for (unsigned p : {53u, 64u, 65u, 127u, 129u, 257u})
        for (unsigned parity : {0u, 1u})
            for (bool negative : {false, true})
                for (bool subtract_small : {false, true}) {
                    point a, expected;
                    a.bit(p);
                    a.words[0] |= parity ? 3 : 1;
                    a.exponent = -std::int64_t(p);
                    a.negative = negative;
                    const auto b = small(1, -gap, negative != subtract_small);
                    expected.bit(p - 1);
                    if (parity) expected.bit(0);
                    if (!subtract_small) expected.increment();
                    expected.exponent = 1 - std::int64_t(p);
                    expected.negative = negative;
                    std::array<std::uint64_t, 8> storage{};
                    sbn::float_buffer out(storage, sbn::bits{p});
                    const auto report = added(out.ref(), a.view(), b.view());
                    equal_point(out.view(), expected);
                    assert(report.error.kind == sbn::error_kind::finite);
                    assert(report.error.exponent2 >= -std::int64_t(p));
                }
}

void specials_comparison_and_range() {
    const auto plus_zero = small(0), minus_zero = small(0, 0, true);
    const auto one = small(1), negative_one = small(1, 0, true);
    const auto inf = sbn::float_view::infinity();
    const auto ninf = sbn::float_view::infinity(true);
    const auto nan = sbn::float_view::nan();
    assert(sbn::compare(nan, one.view()) == std::partial_ordering::unordered);
    assert(sbn::compare(one.view(), nan) == std::partial_ordering::unordered);
    assert(sbn::compare(nan, nan) == std::partial_ordering::unordered);
    assert(sbn::compare(plus_zero.view(), minus_zero.view()) == std::partial_ordering::equivalent);
    assert(sbn::compare(ninf, negative_one.view()) == std::partial_ordering::less);
    assert(sbn::compare(inf, one.view()) == std::partial_ordering::greater);
    assert(sbn::compare(ninf, inf) == std::partial_ordering::less);
    const auto x = small(3, -1), y = small(6, -2);
    assert(sbn::compare(x.view(), y.view()) == std::partial_ordering::equivalent);
    assert(sbn::compare(small(1, -INT64_C(1000000000000)).view(), one.view()) ==
           std::partial_ordering::less);
    assert(sbn::compare(small(1, INT64_C(1000000000000), true).view(), negative_one.view()) ==
           std::partial_ordering::less);

    for (auto mode : modes) {
        std::array<std::uint64_t, 8> storage{};
        sbn::float_buffer out(storage, sbn::bits{65});
        for (bool an : {false, true}) for (bool bn : {false, true})
            for (bool subtract : {false, true}) {
                const auto a = small(0, 0, an), b = small(0, 0, bn);
                const bool effective_bn = bn != subtract;
                const bool sign = an == effective_bn ? an : mode == rounding::downward;
                const auto report = added(out.ref(), a.view(), b.view(), mode, subtract);
                equal_point(out.view(), small(0, 0, sign));
                assert(report.error.exact());
            }
        rounded(out.ref(), minus_zero.view(), mode);
        equal_point(out.view(), minus_zero);
        auto report = added(out.ref(), inf, ninf, mode);
        assert(out.view().is_nan());
        assert(sbn::has_flag(report.flags, sbn::float_flag::invalid));
        report = added(out.ref(), inf, inf, mode, true);
        assert(out.view().is_nan());
        assert(sbn::has_flag(report.flags, sbn::float_flag::invalid));
        added(out.ref(), ninf, one.view(), mode);
        assert(out.view().classification() == sbn::float_class::infinity && out.view().negative());
        added(out.ref(), nan, one.view(), mode);
        assert(out.view().is_nan());
        rounded(out.ref(), inf, mode);
        assert(out.view().classification() == sbn::float_class::infinity && !out.view().negative());

        // Inputs lie inside the documented top-exponent domain. The exact sum
        // is outside it, exercising defined overflow rather than invalid input.
        for (bool negative : {false, true}) {
            const auto maximum = small(1, sbn::float_exponent_limit - 1, negative);
            report = added(out.ref(), maximum.view(), maximum.view(), mode);
            assert(out.view().classification() == sbn::float_class::infinity);
            assert(out.view().negative() == negative);
            assert(sbn::has_flag(report.flags, sbn::float_flag::overflow));
            assert(report.error.kind == sbn::error_kind::unbounded);
            const auto a = small(3, -sbn::float_exponent_limit - 2, negative);
            const auto b = small(2, -sbn::float_exponent_limit - 2, negative);
            report = added(out.ref(), a.view(), b.view(), mode, true);
            equal_point(out.view(), small(0, 0, negative));
            assert(sbn::has_flag(report.flags, sbn::float_flag::underflow));
            assert(report.error.kind == sbn::error_kind::finite);
            assert(report.error.exponent2 >= -sbn::float_exponent_limit - 2);
        }
    }
}

void owner_precision_and_capacity() {
    alignas(128) std::array<std::byte, 8192> arena{};
    sbn::buffer_resource resource(arena);
    {
        sbn::bigfloat x(sbn::bits{65}, resource);
        const auto exact = small(11, -3);
        auto report = x.assign(exact.view());
        assert(report.error.exact());
        x.reserve_bits(sbn::bits{513});
        assert(x.precision().count == 65 && x.capacity() >= 9);
        equal_point(x.view(), exact);
        const auto before = resource.statistics();
        allocation_watch_start();
        report = x.set_precision(sbn::bits{3}, rounding::nearest_even);
        assert(allocation_watch_stop() == 0);
        equal_point(x.view(), small(6, -2)); // 11/8 halfway -> 12/8, even significand.
        error_at_scale(report, 1, -3);
        allocation_watch_start();
        report = x.set_precision(sbn::bits{129});
        assert(allocation_watch_stop() == 0);
        assert(report.error.exact() && x.precision().count == 129);
        equal_point(x.view(), small(6, -2));
        const auto after = resource.statistics();
        assert(before.live_bytes == after.live_bytes);
        assert(before.live_allocations == after.live_allocations);
        auto copy = x;
        equal_point(copy.view(), small(6, -2));
        assert(copy.capacity() >= (copy.precision().count + 63) / 64);
        report = rounded(copy.ref(), copy.view());
        assert(report.error.exact());
        sbn::bigfloat assigned(sbn::bits{1}, resource);
        assigned = x;
        assert(assigned.precision().count == 129);
        assert(assigned.capacity() >= 3);
        report = rounded(assigned.ref(), assigned.view());
        assert(report.error.exact());
        auto moved = std::move(copy);
        equal_point(moved.view(), small(6, -2));
        assert(moved.precision().count == 129);
    }
    assert(resource.statistics().live_allocations == 0);
    std::array<std::uint64_t, 1> too_small{};
    sbn::float_buffer out(too_small, sbn::bits{129});
    const auto a = small(1), b = small(3);
    allocation_watch_start();
    const auto need = sbn::add_requirements(out.ref(), a.view(), b.view());
    assert(allocation_watch_stop() == 0);
    assert(need.status == sbn::support::capacity && need.output_limbs == 3);
}
} // namespace

int main() {
    small_exact_oracle();
    multi_limb_rounding_witnesses();
    cancellation_and_aliases();
    virtual_exponent_gaps();
    specials_comparison_and_range();
    owner_precision_and_capacity();
    {
        std::array<std::uint64_t,3> words{};sbn::float_buffer out(words,sbn::bits{129});
        const auto x=small(7,-5,true);
        allocation_watch_start();auto r=sbn::scale2_into(out.ref(),x.view(),1000000000000LL);
        assert(!allocation_watch_stop());assert(r.error.exact());
        equal_point(out.view(),small(7,999999999995LL,true));
        r=sbn::scale2_into(out.ref(),out.view(),-1000000000000LL);equal_point(out.view(),x);assert(r.error.exact());
        r=sbn::scale2_into(out.ref(),x.view(),INT64_MAX);assert(!out.view().finite()&&out.view().negative());
        assert(sbn::has_flag(r.flags,sbn::float_flag::overflow)&&r.error.kind==sbn::error_kind::unbounded);
        r=sbn::scale2_into(out.ref(),x.view(),INT64_MIN);assert(out.view().zero()&&out.view().negative());
        assert(sbn::has_flag(r.flags,sbn::float_flag::underflow)&&!r.error.exact());
        auto a=sbn::bigfloat::from(0.1);equal_point(a.view(),small(7205759403792794ULL,-56));
        auto b=sbn::bigfloat::from(-0.0);equal_point(b.view(),small(0,0,true));
        auto c=sbn::bigfloat::from(std::numeric_limits<double>::denorm_min());equal_point(c.view(),small(1,-1074));
        auto d=sbn::bigfloat::from(std::numeric_limits<double>::infinity());assert(d.view().classification()==sbn::float_class::infinity);
        auto e=sbn::bigfloat::from(std::numeric_limits<double>::quiet_NaN());assert(e.view().is_nan());
        sbn::float_buffer tiny(words,sbn::bits{1});
        const auto a0=small(7,-sbn::float_exponent_limit-3),b0=small(4,-sbn::float_exponent_limit-3);
        const auto three=small(3);
        r=sbn::scale2_into(tiny.ref(),three.view(),-sbn::float_exponent_limit-3);
        equal_point(tiny.view(),small(1,-sbn::float_exponent_limit-1));
        assert(!sbn::has_flag(r.flags,sbn::float_flag::underflow));
        r=added(tiny.ref(),a0.view(),b0.view(),rounding::nearest_even,true);
        equal_point(tiny.view(),small(1,-sbn::float_exponent_limit-1));
        assert(!sbn::has_flag(r.flags,sbn::float_flag::underflow));
    }
    std::printf("float core: %zu admitted operations, exact dyadic/limb witnesses, "
                "rounding, aliases, virtual gaps, reports and specials PASS\n",
                checked_operations);
}
