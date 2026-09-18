#include "series/formulas.hpp"
#include "common/checked.hpp"
#include "core/x86_64/word.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
extern "C" uint64_t sbn3i_mul_1c(uint64_t *, const uint64_t *, long, uint64_t, uint64_t);

namespace sbn::v3::series {
namespace {
constexpr uint64_t A = 13591409, B = 545140134, K = 10939058860032000ULL;
static_assert(__uint128_t(640320) * 640320 * 640320 / 24 == K);
unsigned bits(uint64_t x) {
    return x ? 64 - __builtin_clzll(x) : 0;
}
unsigned bits(__uint128_t x) {
    return x >> 64 ? 64 + bits(uint64_t(x >> 64)) : bits(uint64_t(x));
}
unsigned ceil_log(uint64_t n) {
    return n > 1 ? bits(n - 1) : 0;
}
size_t words(__uint128_t b) {
    return size_t((b + 63) / 64);
}
void set(sbn3_series_value &v, const uint64_t *a, size_t n, bool negative = false, int64_t exponent = 0) {
    while (n && !a[n - 1])
        --n;
    require(v.mantissa.data && v.mantissa.capacity >= n, SBN3_FATAL_WORKSPACE, "series leaf capacity", n,
            v.mantissa.capacity);
    if (n)
        std::memcpy(v.mantissa.data, a, n * 8);
    v.mantissa.size = n;
    v.mantissa.negative = n && negative;
    v.exponent2 = exponent;
}
size_t times(uint64_t *v, size_t n, uint64_t x) {
    const uint64_t carry = sbn3i_mul_1(v, v, long(n), x);
    if (carry)
        v[n++] = carry;
    return n;
}
} // namespace
__uint128_t integer_log_sum(uint64_t a, uint64_t b) noexcept {
    require(a >= 1 && a < b && b <= uint64_t(1) << 48, SBN3_FATAL_ARGUMENT, "integer log interval");
    __uint128_t sum = 0;
    while (a < b) {
        const unsigned width = bits(a);
        const uint64_t next = std::min(b, uint64_t(1) << width);
        sum += __uint128_t(next - a) * width;
        a = next;
    }
    return sum;
}
sbn3_series_recipe Formula::recipe() const noexcept {
    return kind == FormulaKind::Euler        ? SBN3_SERIES_HYPERDESCENT
           : kind == FormulaKind::Chudnovsky ? SBN3_SERIES_COMMON_P2B3
                                             : SBN3_SERIES_BINARY_BBP;
}
sbn3_query_result Formula::bounds(sbn3_series_range r, uint64_t n, unsigned need,
                                  sbn3_series_shape &shape, bool normalized) const noexcept {
    shape = {};
    if (r.begin >= r.end || r.end - 1 > max_index || !n || n > r.end - r.begin ||
        (kind != FormulaKind::Chudnovsky && !r.begin) || !need ||
        (need & ~(kind == FormulaKind::Chudnovsky ? 7u : 3u)) ||
        (kind == FormulaKind::BinaryLog && (!radix_bits || radix_bits > 32767)))
        return SBN3_UNSUPPORTED;
    // Monotonically increasing factor bounds attain their maximum in the last
    // n terms of the envelope. k=0 has D=U=1 and is accounted separately.
    const uint64_t a = std::max(uint64_t(1), r.end - n), m = r.end - a;
    const __uint128_t q = m ? integer_log_sum(a, r.end) : 0;
    __uint128_t tb = 0, db = 0, ub = 0;
    if (kind == FormulaKind::Chudnovsky) {
        // D=K^m product(k^3), U=product((6k-5)(2k-1)(6k-1)).
        // K<2^54, factors' product<72*k^3<2^7*k^3; |U/D|<=1.
        // Thus |T/D|<=n*(A+B*(end-1)), also for the special k=0 leaf.
        db = 54 * __uint128_t(m) + 3 * q + 1;
        ub = 7 * __uint128_t(m) + 3 * q + 1;
        tb = db + ceil_log(n) + bits(__uint128_t(B) * (r.end - 1) + A);
        // Every positive-index Q factor contains 2^15. Canonical mantissas
        // omit whole low zero words; +63 covers their remaining partial word.
        // The per-factor raw bit bound minus 15 is positive/monotone, so
        // this covers all shorter subranges too. Truncating positive Q can
        // only increase its 2-adic valuation and decrease its magnitude.
        if(normalized && m)db=std::min(db,db-15*__uint128_t(m)+63);
    } else if (kind == FormulaKind::Euler) {
        db = __uint128_t(factorial_log_bounds(a - 1, r.end - 1).upper) + 1;
        tb = db + 1; // positive-start local Euler ratio is below 2
    } else {
        db = q + 1;
        // BinaryLog's T is exact at exponent -r*(last index).
        tb = db + __uint128_t(radix_bits) * (n - 1) + ceil_log(n) + 1;
    }
    const __uint128_t limit = __uint128_t(SIZE_MAX / 8) * 64;
    if (std::max({tb, db, ub}) > limit)
        return SBN3_QUERY_CAPACITY;
    const __uint128_t sizes[]{tb, db, ub};
    for (unsigned j = 0; j < 3; ++j)
        if (need & (1u << j))
            shape.limbs[j] = std::max(size_t(1), words(sizes[j]));
    return SBN3_SUPPORTED;
}
double Formula::work(sbn3_series_range r) const noexcept {
    // Additive logarithmic mass, with exact bucket sums before conversion.
    // It estimates relative work, not elapsed time or the full live memory.
    const uint64_t a = std::max(uint64_t(1), r.begin), m = r.end - a;
    const __uint128_t q = m ? integer_log_sum(a, r.end) : 0;
    if (kind == FormulaKind::Chudnovsky)
        return double(54 * __uint128_t(m) + 3 * q + (r.begin == 0));
    return double(q + (r.end - r.begin));
}
namespace {
constexpr double log2e = 1.4426950408889634074;
constexpr double logK = 36.931116160784572;
uint64_t roundoff_bits(double mass) {
    // Generous allowance for the stable log1p increment, products, division
    // by ln(2), and split comparisons. At the supported maximum indices this
    // stays bounded; at 5B it rounds to one bit plus two guard bits.
    return uint64_t(std::ceil(64 * std::numeric_limits<double>::epsilon() * (mass + 1))) + 2;
}
}
FactorialLogBounds factorial_log_bounds(uint64_t a, uint64_t b) noexcept {
    require(a <= b && b <= (uint64_t(1) << 48), SBN3_FATAL_ARGUMENT, "factorial log interval");
    if (a == b || b <= 1)
        return {};
    const double h = double(b - a);
    const double estimate = (h * std::log1p(double(a)) +
                             (double(b) + .5) * std::log1p(h / (double(a) + 1)) - h) * log2e;
    const double guard = double(roundoff_bits(estimate));
    const double lower = std::floor(estimate - .125 * log2e - guard);
    return {lower > 0 ? uint64_t(lower) : 0, uint64_t(std::ceil(estimate + guard))};
}
double Formula::split_mass(sbn3_series_range r) const noexcept {
    require(kind == FormulaKind::Chudnovsky && r.begin < r.end && r.end - 1 <= max_index,
            SBN3_FATAL_ARGUMENT, "Chudnovsky split domain");
    // Positive terms are k=max(1,begin)..end-1; the k=0 leaf gets one
    // artificial bit of mass. This is additive and avoids a zero-size leaf.
    const uint64_t a = std::max(uint64_t(1), r.begin) - 1, b = r.end - 1;
    const double h = double(b - a);
    // Stable g(b)-g(a), g(x)=(x+1/2)ln(1+x)-x. Do not subtract two
    // enormous log-factorial prefixes for a short interval at a large index.
    const double mass = h * (logK + 3 * std::log1p(double(a)) - 3) +
                        3 * (double(b) + .5) * std::log1p(h / (double(a) + 1));
    return mass * log2e + (r.begin == 0);
}
uint64_t Formula::split_point(sbn3_series_range r, double fraction) const noexcept {
    const uint64_t length = r.end - r.begin;
    require(length >= 2 && fraction > 0 && fraction < 1, SBN3_FATAL_ARGUMENT, "series split fraction");
    const uint64_t margin = std::max(uint64_t(1), length / 4);
    // Far from zero the log-density changes too little across this interval
    // to move the nearest equal-mass cut beyond the arithmetic midpoint.
    // For odd lengths, increasing density favors the longer LEFT half.
    if (fraction == .5 && __uint128_t(length) * length <= r.begin)
        return r.begin + (length + 1) / 2;
    const double target = split_mass(r) * fraction;
    uint64_t lo = r.begin + 1, hi = r.end - 1;
    while (lo < hi) {
        const uint64_t m = lo + (hi - lo) / 2;
        if (split_mass({r.begin, m}) < target)
            lo = m + 1;
        else
            hi = m;
    }
    if (lo > r.begin + 1 && std::abs(split_mass({r.begin, lo - 1}) - target) <
                                std::abs(split_mass({r.begin, lo}) - target))
        --lo;
    return std::clamp(lo, r.begin + margin, r.end - margin);
}
sbn3_query_result Formula::serial_envelope(sbn3_series_range r, unsigned depth, unsigned need,
                                          uint64_t &max_terms, sbn3_series_shape &shape, bool normalized) const noexcept {
    if (kind != FormulaKind::Chudnovsky || r.begin >= r.end || r.end - 1 > max_index ||
        !need || (need & ~7u) || depth > 96)
        return SBN3_UNSUPPORTED;
    const double estimate = split_mass(r);
    uint64_t mass = uint64_t(std::ceil(estimate)) + roundoff_bits(estimate);
    const uint64_t high = 54 + 3 * bits(std::max(uint64_t(1), r.end - 1));
    const uint64_t low = 53 + 3 * (bits(std::max(uint64_t(1), r.begin)) - 1);
    max_terms = r.end - r.begin;
    for (unsigned j = 0; j < depth; ++j) {
        // Integral/trapezoid error in a Q interval is <3/(8 ln 2) bits.
        // Rounding a balanced cut adds at most half one term's mass.
        mass = (mass + 1) / 2 + (high + 1) / 2 + roundoff_bits(double(mass)) + 2;
        const uint64_t clamped = max_terms > 1 ? max_terms - std::max(uint64_t(1), max_terms / 4) : 1;
        max_terms = std::max(uint64_t(1), std::min(clamped, mass / low + (r.begin == 0)));
    }
    // log2(U/Q) < -47*m and log2(Q) <= high*m for m positive-index
    // terms, hence log2(U) <= mass*(high-47)/high. No need to combine
    // the largest term count at the left with the largest factors at the right.
    const __uint128_t raw_db = __uint128_t(mass) + 1;
    // If log2(Q)<=min(mass,high*k), then log2(Q)-15*k is at most
    // mass*(high-15)/high. This does not assume every descendant has the
    // envelope's maximum term count. Keep a whole-word normalization guard.
    const __uint128_t db=normalized?std::min(raw_db,(__uint128_t(mass)*(high-15)+high-1)/high+64):raw_db;
    const __uint128_t ub = (__uint128_t(mass) * (high - 47) + high - 1) / high + 1;
    const __uint128_t tb = raw_db + ceil_log(max_terms) + bits(__uint128_t(B) * (r.end - 1) + A);
    const __uint128_t value_bits[]{tb, db, ub};
    shape = {};
    for (unsigned j = 0; j < 3; ++j)
        if (need & (1u << j))
            shape.limbs[j] = std::max(size_t(1), words(value_bits[j]));
    return SBN3_SUPPORTED;
}
void Formula::leaf(uint64_t k, unsigned need, sbn3_series_values &out) const noexcept {
    require(k <= max_index && (kind == FormulaKind::Chudnovsky || k >= 1), SBN3_FATAL_ARGUMENT,
            "series leaf index");
    if (kind != FormulaKind::Chudnovsky) {
        const uint64_t one = 1;
        require(kind != FormulaKind::BinaryLog || (radix_bits && radix_bits <= 32767), SBN3_FATAL_ARGUMENT,
                "BBP exponent domain");
        if (need & 1)
            set(out.value[0], &one, 1, false, kind == FormulaKind::BinaryLog ? -int64_t(k * radix_bits) : 0);
        if (need & 2)
            set(out.value[1], &k, 1);
        return;
    }
    if (!k) {
        const uint64_t one = 1;
        if (need & 1)
            set(out.value[0], &A, 1);
        if (need & 2)
            set(out.value[1], &one, 1);
        if (need & 4)
            set(out.value[2], &one, 1);
        return;
    }
    uint64_t p[5]{6 * k - 5}, d[5]{K};
    size_t pn = 1, dn = 1;
    if (need & 5) {
        pn = times(p, pn, 2 * k - 1);
        pn = times(p, pn, 6 * k - 1);
    }
    if (need & 4)
        set(out.value[2], p, pn);
    if (need & 2) {
        for (unsigned j = 0; j < 3; ++j)
            dn = times(d, dn, k);
        set(out.value[1], d, dn);
    }
    if (need & 1) {
        const __uint128_t factor = __uint128_t(B) * k + A;
        uint64_t f[2]{uint64_t(factor), uint64_t(factor >> 64)}, t[7]{};
        const size_t fn = f[1] ? 2 : 1;
        sbn3_mul_basecase(t, pn + fn, p, pn, f, fn);
        set(out.value[0], t, pn + fn, k & 1);
    }
}
void Formula::euler_batch(sbn3_series_range r, unsigned need, sbn3_series_values &out) const noexcept {
    require(kind == FormulaKind::Euler && r.begin >= 1 && r.begin < r.end && r.end - 1 <= max_index,
            SBN3_FATAL_ARGUMENT, "Euler batch interval");
    for (unsigned j = 0; j < 2; ++j)
        if (need & (1u << j)) {
            auto &v = out.value[j];
            require(v.mantissa.capacity, SBN3_FATAL_WORKSPACE, "Euler batch output");
            v.mantissa.size = 1;
            v.mantissa.negative = 0;
            v.exponent2 = 0;
            v.mantissa.data[0] = j ? 1 : 0;
        }
    for (uint64_t k = r.begin; k < r.end; ++k)
        for (unsigned j = 0; j < 2; ++j)
            if (need & (1u << j)) {
                auto &v = out.value[j].mantissa;
                const uint64_t carry = sbn3i_mul_1c(v.data, v.data, long(v.size), k, j ? 0 : 1);
                if (carry) {
                    require(v.size < v.capacity, SBN3_FATAL_MATH, "Euler batch bound");
                    v.data[v.size++] = carry;
                }
            }
}
} // namespace sbn::v3::series
