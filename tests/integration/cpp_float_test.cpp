// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/sbn.hpp"
#include <array>
#include <cassert>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();

namespace {
using sbn::bigfloat;
using u128 = __uint128_t;

template<class T> concept HasRvalueView = requires(T&& x) { std::move(x).view(); };
static_assert(!HasRvalueView<bigfloat>);
static_assert(std::same_as<decltype(std::declval<const bigfloat&>() +
                                  std::declval<const bigfloat&>()), bigfloat>);
static_assert(std::same_as<decltype(std::declval<const bigfloat&>() <=>
                                  std::declval<const bigfloat&>()), std::partial_ordering>);

struct RecordingResource final : sbn::memory_resource {
    sbn::buffer_resource storage;
    unsigned allocations = 0, deallocations = 0;
    explicit RecordingResource(std::span<std::byte> bytes) noexcept : storage(bytes) {}
    sbn::resource_error try_allocate(std::size_t n, std::size_t alignment,
                                    sbn::allocation& out) noexcept override {
        ++allocations;
        return storage.try_allocate(n, alignment, out);
    }
    void deallocate(sbn::allocation block) noexcept override {
        // Count empty calls too: an unnecessary zero-scratch workspace has
        // an observable destructor call even when it performs no OS allocation.
        ++deallocations;
        storage.deallocate(block);
    }
    sbn::resource_statistics statistics() const noexcept override { return storage.statistics(); }
};
void assign_integer(bigfloat& x, std::int64_t value) {
    const bool negative = value < 0;
    const auto u = std::uint64_t(value);
    const std::uint64_t m = negative ? std::uint64_t(0) - u : u;
    const auto report = x.assign(sbn::float_view::from_magnitude({&m, 1}, 0, negative));
    assert(report.error.exact());
}
void same_point(sbn::float_view a, sbn::float_view b) {
    assert(a.classification() == b.classification());
    if (a.is_nan()) return;
    assert(sbn::compare(a, b) == std::partial_ordering::equivalent);
    if (a.zero()) assert(a.negative() == b.negative());
}
void small_eager_paths() {
    for (int a = -12; a <= 12; ++a) for (int b = -12; b <= 12; ++b) {
        allocation_watch_start();
        {
            const auto x = bigfloat::from(a, sbn::bits{53});
            const auto y = bigfloat::from(b, sbn::bits{65});
            const auto sum = x + y, difference = x - y, product = x * y;
            assert(sum.precision().count == 65 && difference.precision().count == 65);
            assert(product.precision().count == 65);
            assert(sum == bigfloat::from(a + b, sbn::bits{65}));
            assert(difference == bigfloat::from(a - b, sbn::bits{65}));
            assert(product == bigfloat::from(a * b, sbn::bits{65}));
            assert(sbn::add(x, y) == sum);
            assert(sbn::subtract(x, y) == difference);
            assert(sbn::multiply(x, y) == product);
            const auto sq = sbn::square(x);
            assert(sq.precision().count == 53 && sq == bigfloat::from(a * a, sbn::bits{53}));
            const auto positive = +x, negative = -x;
            assert(positive == x && positive.precision().count == 53);
            assert(negative == bigfloat::from(-a, sbn::bits{53}));
            assert((x == y) == (a == b));
            assert(((x <=> y) < 0) == (a < b));
            assert(((x <=> y) > 0) == (a > b));

            auto in_place = x;
            assert(&(in_place += y) == &in_place);
            assert(in_place == sum && in_place.precision().count == 65);
            assert(&(in_place -= y) == &in_place);
            assert(in_place == x && in_place.precision().count == 65);
            assert(&(in_place *= y) == &in_place);
            assert(in_place == product);
            auto self = x;
            self += self;
            assert(self == bigfloat::from(2 * a, sbn::bits{53}));
            self -= self;
            assert(self.view().zero() && !self.view().negative());
            self = x;
            self *= self;
            assert(self == sq);
        }
        assert(allocation_watch_stop() == 0);
    }
    // A previous approximation is a stored point, not a hidden radius.
    const auto rounded_three = bigfloat::from(3, sbn::bits{1}); // stored 4
    const auto one = bigfloat::from(1, sbn::bits{2});
    assert((rounded_three - one) == bigfloat::from(3, sbn::bits{2}));
    assert((rounded_three - one).precision().count == 2);
}

void custom_resources_and_lifetimes() {
    alignas(4096) std::array<std::byte, 262144> left_bytes{}, right_bytes{};
    RecordingResource left(left_bytes), right(right_bytes);
    {
        bigfloat a(sbn::bits{53}, left), b(sbn::bits{65}, right);
        assign_integer(a, 4);
        assign_integer(b, 3);
        const auto la = left.allocations, ld = left.deallocations;
        const auto ra = right.allocations, rd = right.deallocations;
        allocation_watch_start();
        {
            auto sum = a + b, difference = a - b, product = a * b;
            auto sq = sbn::square(a), unary = -a;
            auto root = sbn::sqrt(a), inverse = sbn::rsqrt(a, sbn::bits{113});
            assert(&sum.resource() == &left && &difference.resource() == &left);
            assert(&product.resource() == &left && &sq.resource() == &left);
            assert(&unary.resource() == &left && &root.resource() == &left);
            assert(&inverse.resource() == &left);
            assert(root == bigfloat::from(2, sbn::bits{53}));
            assert(inverse == bigfloat::from(0.5, sbn::bits{113}));
            auto reverse = b + a;
            assert(&reverse.resource() == &right && reverse.precision().count == 65);
            a += b;
            a -= b;
            a *= b;
            assert(&a.resource() == &left && a == bigfloat::from(12, sbn::bits{65}));
        }
        assert(allocation_watch_stop() == 0);
        assert(left.allocations == la && left.deallocations == ld);
        assert(right.allocations == ra && right.deallocations == rd);

        bigfloat high_precision(sbn::bits{513}, right);
        assign_integer(high_precision, 3);
        // The output needs owned storage, but this short-value multiplication
        // has zero scratch. Only its value allocation is permitted.
        const auto before_alloc = left.allocations, before_free = left.deallocations;
        {
            auto product = a * high_precision;
            assert(product.precision().count == 513 && &product.resource() == &left);
            assert(product == bigfloat::from(36, sbn::bits{513}));
            assert(left.allocations == before_alloc + 1);
            assert(left.deallocations == before_free);
        }
        assert(left.deallocations == before_free + 1);

        assign_integer(a, 2);
        const auto before_live = left.statistics().live_allocations;
        const auto before_root_alloc = left.allocations;
        const auto right_alloc = right.allocations;
        {
            auto root = sbn::sqrt(a, sbn::bits{257});
            assert(root.precision().count == 257 && &root.resource() == &left);
            assert(left.allocations >= before_root_alloc + 2); // value + nonzero scratch
            assert(left.statistics().live_allocations == before_live + 1);
            assert(right.allocations == right_alloc);
            auto copy = root;
            assert(copy == root && &copy.resource() == &left);
            assert(copy.capacity() >= 5);
            auto moved = std::move(copy);
            assert(moved == root && moved.precision().count == 257);
            assert(copy.view().zero() && copy.precision().count == 128);
            bigfloat assigned(sbn::bits{1}, right);
            assigned = moved;
            assert(assigned == root && assigned.precision().count == 257);
            assert(&assigned.resource() == &right && assigned.capacity() >= 5);
            assigned = std::move(moved);
            assert(assigned == root && &assigned.resource() == &right);
            assert(moved.view().zero() && moved.precision().count == 128);
        }
        assert(left.statistics().live_allocations == before_live);

        // Compound operations may grow the owning destination at this explicit
        // convenience boundary; they retain its provider and handle self-alias.
        a += high_precision;
        assert(a.precision().count == 513 && &a.resource() == &left);
        assert(a == bigfloat::from(5, sbn::bits{513}));
        a *= a;
        assert(a == bigfloat::from(25, sbn::bits{513}));
        a -= a;
        assert(a.view().zero() && a.precision().count == 513);
    }
    assert(left.statistics().live_allocations == 0);
    assert(right.statistics().live_allocations == 0);
}

void root_interval_witness(sbn::float_view value, sbn::error_bound error,
                           unsigned radicand, bool inverse) {
    // For these <=53-bit tests, integer endpoint squares fit in 128 bits.
    // This checks the claimed error, not equality with a correctly rounded root.
    assert(value.finite() && !value.negative() && !value.zero());
    assert(value.magnitude().size() == 1);
    assert(error.kind == sbn::error_kind::finite);
    const auto base = value.exponent2() < error.exponent2
                    ? value.exponent2() : error.exponent2;
    const auto value_shift = value.exponent2() - base;
    const auto error_shift = error.exponent2 - base;
    assert(value_shift >= 0 && value_shift < 64);
    assert(error_shift >= 0 && error_shift < 64);
    assert(base < 0 && -base <= 60);
    const u128 middle = u128(value.magnitude().data()[0]) << unsigned(value_shift);
    const u128 radius = u128(1) << unsigned(error_shift);
    assert(middle >= radius);
    const u128 lower = middle - radius, upper = middle + radius;
    assert((upper >> 60) == 0);
    const auto unit = u128(1) << unsigned(-2 * base);
    if (inverse) {
        assert(lower * lower * radicand <= unit);
        assert(upper * upper * radicand >= unit);
    } else {
        assert(lower * lower <= u128(radicand) * unit);
        assert(upper * upper >= u128(radicand) * unit);
    }
}
void root_conveniences_match_controlled() {
    for (unsigned p : {3u, 17u, 53u, 65u, 129u, 257u}) {
        const auto x = bigfloat::from(2, sbn::bits{65});
        for (bool inverse : {false, true}) {
            const auto eager = inverse ? sbn::rsqrt(x, sbn::bits{p})
                                       : sbn::sqrt(x, sbn::bits{p});
            assert(eager.precision().count == p);
            std::array<std::uint64_t, 8> words{};
            sbn::float_buffer controlled(words, sbn::bits{p});
            const auto need = inverse
                ? sbn::rsqrt_requirements(controlled.ref(), x.view())
                : sbn::sqrt_requirements(controlled.ref(), x.view());
            assert(need.supported());
            sbn::workspace work;
            work.reserve(need.scratch_bytes, need.scratch_alignment);
            allocation_watch_start();
            const auto report = inverse
                ? sbn::rsqrt_into(controlled.ref(), x.view(), work.on())
                : sbn::sqrt_into(controlled.ref(), x.view(), work.on());
            assert(allocation_watch_stop() == 0);
            same_point(eager.view(), controlled.view());
            if (p <= 53) root_interval_witness(eager.view(), report.error, 2, inverse);
        }
    }
    const auto x = bigfloat::from(2, sbn::bits{129});
    assert(sbn::sqrt(x).precision().count == 129);
    assert(sbn::rsqrt(x).precision().count == 129);
    for (unsigned p : {1u, 53u, 65u, 129u, 257u}) {
        const auto four = bigfloat::from(4, sbn::bits{p});
        assert(sbn::sqrt(four) == bigfloat::from(2, sbn::bits{p}));
        assert(sbn::rsqrt(four) == bigfloat::from(0.5, sbn::bits{p}));
    }
}
void exceptional_points_and_ordering() {
    auto zero = bigfloat::from(0, sbn::bits{65});
    const auto negative_zero = -zero;
    assert(zero == negative_zero && (zero <=> negative_zero) == std::partial_ordering::equivalent);
    assert(negative_zero.view().negative());
    const auto twice_negated = -negative_zero;
    const auto unary_plus = +negative_zero;
    const auto root_zero = sbn::sqrt(negative_zero);
    assert(twice_negated.view().zero() && !twice_negated.view().negative());
    assert(unary_plus.view().negative());
    assert(root_zero.view().negative());
    const auto inverse_zero = sbn::rsqrt(negative_zero);
    assert(inverse_zero.view().classification() == sbn::float_class::infinity);
    assert(inverse_zero.view().negative());

    bigfloat inf(sbn::bits{53}), nan(sbn::bits{53});
    inf.assign(sbn::float_view::infinity());
    nan.assign(sbn::float_view::nan());
    assert(nan != nan && !(nan == zero));
    assert((nan <=> zero) == std::partial_ordering::unordered);
    assert(!(nan < zero) && !(nan > zero) && !(nan <= zero) && !(nan >= zero));
    assert(inf > zero && -inf < zero);
    const auto indeterminate_difference = inf - inf;
    const auto indeterminate_product = zero * inf;
    const auto invalid_root = sbn::sqrt(bigfloat::from(-1));
    assert(indeterminate_difference.view().is_nan());
    assert(indeterminate_product.view().is_nan());
    assert(invalid_root.view().is_nan());
}
void power_and_quotient_conveniences(){
    const auto two=bigfloat::from(2,sbn::bits{129});
    assert(sbn::powi(two,3)==bigfloat::from(8));
    assert(sbn::powi(two,-3)==bigfloat::from(0.125));
    assert(sbn::powi(two,0,sbn::bits{17}).precision().count==17);
    assert(sbn::rootn(bigfloat::from(8),3)==two);
    assert(sbn::rootn(bigfloat::from(-8),3)==-two);
    assert(sbn::rootn(two,1,sbn::bits{65}).precision().count==65);
    const auto three=bigfloat::from(3,sbn::bits{65});
    assert(three/two==bigfloat::from(1.5));
    assert(sbn::reciprocal(two)==bigfloat::from(0.5));
    assert(sbn::agm(two,two)==two);
    auto value=three;value/=two;assert(value==bigfloat::from(1.5));
    const auto fraction=bigfloat::from(1.75,sbn::bits{65});
    assert(sbn::to_integer(fraction)==sbn::integer{2});
    assert(sbn::to_integer(fraction,sbn::rounding::downward)==sbn::integer{1});
    assert(sbn::to_integer(-fraction,sbn::rounding::upward)==sbn::integer{-1});
}
void elementary_conveniences(){
    // The numerical suites own rigorous high-precision accuracy checks.
    // Here exercise public owner wiring, exact leaves and paired outputs.
    using Unary=bigfloat(*)(const bigfloat&)noexcept;
    struct Case {Unary function;int input,expected;};
    const Case cases[]{
        {sbn::exp,0,1},{sbn::expm1,0,0},{sbn::log,1,0},{sbn::log1p,0,0},
        {sbn::sin,0,0},{sbn::cos,0,1},{sbn::sinpi,1,0},{sbn::cospi,1,-1},
        {sbn::atan,0,0},{sbn::asin,0,0},{sbn::acos,1,0},
        {sbn::sinh,0,0},{sbn::cosh,0,1},{sbn::tanh,0,0},
        {sbn::asinh,0,0},{sbn::acosh,1,0},{sbn::atanh,0,0},
        {sbn::exp2,3,8},{sbn::log2,8,3},{sbn::log10,1,0}};
    for(const auto& c:cases){
        const auto input=bigfloat::from(c.input,sbn::bits{65});
        const auto output=c.function(input);
        assert(output.precision().count==65&&output==bigfloat::from(c.expected));
    }
    const auto one=bigfloat::from(1,sbn::bits{65});
    const auto two=bigfloat::from(2,sbn::bits{65});
    const auto three=bigfloat::from(3,sbn::bits{65});
    assert(sbn::exp(one)>two&&sbn::exp(one)<three);
    assert(sbn::log(two)>bigfloat::from(0.5)&&sbn::log(two)<one);
    const auto a=sbn::atan2(one,one);
    assert(a.precision().count==65&&a>bigfloat::from(0.5)&&a<one);
    assert(sbn::pow(two,three)==bigfloat::from(8));
    const auto pair=sbn::sincos(bigfloat::from(0,sbn::bits{65}));
    assert(pair.sine.view().zero()&&pair.cosine==one);
    const auto quarter=sbn::sincospi(bigfloat::from(0.25,sbn::bits{65}));
    assert(quarter.sine>bigfloat::from(0.5)&&quarter.sine<one);
    assert(quarter.cosine>bigfloat::from(0.5)&&quarter.cosine<one);
    const auto elliptic=sbn::elliptic_ke(bigfloat::from(0,sbn::bits{65}));
    assert(elliptic.k>one&&elliptic.k<two&&elliptic.e>one&&elliptic.e<two);
    assert(elliptic.k.precision().count==65&&elliptic.e.precision().count==65);
}
} // namespace

int main() {
    small_eager_paths();
    custom_resources_and_lifetimes();
    root_conveniences_match_controlled();
    exceptional_points_and_ordering();
    power_and_quotient_conveniences();
    elementary_conveniences();
    std::puts("C++ bigfloat: eager operators, precision/resources, zero-scratch leaves, "
              "copy/move and bounded-root witnesses PASS");
}
