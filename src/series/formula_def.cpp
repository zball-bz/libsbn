#include "series/formula_def.hpp"
#include "series/log_product.hpp"
#include "sbn3/value.h"
#include "common/checked.hpp"
#include "common/identity.hpp"
#include "core/x86_64/word.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
extern "C" uint64_t sbn3i_mul_1c(uint64_t *, const uint64_t *, long, uint64_t, uint64_t);

namespace sbn::v3::series {
namespace {
constexpr uint64_t max_index = (uint64_t(1) << 48) - 1;
constexpr unsigned max_leaf_words = 64; // must agree with arithmetic.cpp
using u128 = __uint128_t;
using i128 = __int128_t;
unsigned bits(uint64_t x) {
    return x ? 64 - __builtin_clzll(x) : 0;
}
unsigned bits(u128 x) {
    return x >> 64 ? 64 + bits(uint64_t(x >> 64)) : bits(uint64_t(x));
}
unsigned ceil_log(uint64_t n) {
    return n > 1 ? bits(n - 1) : 0;
}
size_t words(u128 b) {
    return size_t((b + 63) / 64);
}
bool power_of_two(u128 x) {
    return x && !(x & (x - 1));
}
u128 constant(const FactorProduct &f) {
    return (u128(f.constant_high) << 64) | f.constant_low;
}
// Signed exact value of a factor at k (|a*k+b| < 2^113 for k < 2^48).
i128 factor_value(const LinearFactor &f, uint64_t k) {
    return i128(u128(f.a) * k) + i128(f.b);
}
void set(sbn3_series_value &v, const uint64_t *a, size_t n, bool negative, int64_t exponent) {
    while (n && !a[n - 1])
        --n;
    require(v.mantissa.data && v.mantissa.capacity >= n, SBN3_FATAL_WORKSPACE, "formula leaf capacity", n,
            v.mantissa.capacity);
    if (n)
        std::memcpy(v.mantissa.data, a, n * 8);
    v.mantissa.size = n;
    v.mantissa.negative = n && negative;
    v.exponent2 = exponent;
}
size_t times(uint64_t *v, size_t n, uint64_t x) {
    if (!n)
        return 0;
    const uint64_t carry = sbn3i_mul_1(v, v, long(n), x);
    if (carry)
        v[n++] = carry;
    return n;
}
// v <- v * (hi*2^64 + lo), hi != 0. Two word passes; no allocation.
size_t times2(uint64_t *v, size_t n, uint64_t lo, uint64_t hi) {
    if (!n)
        return 0;
    uint64_t t[max_leaf_words + 2];
    require(n + 2 <= max_leaf_words + 2, SBN3_FATAL_WORKSPACE, "formula factor width");
    t[n] = sbn3i_mul_1(t, v, long(n), lo);
    t[n + 1] = sbn3i_addmul_1(t + 1, v, long(n), hi);
    size_t m = n + 2;
    while (m && !t[m - 1])
        --m;
    std::memcpy(v, t, m * 8);
    return m;
}
// Exact product value at k into w (capacity max_leaf_words): returns words,
// zero words for a zero value. Sign includes the constant and (-1)^k.
size_t evaluate(const FactorProduct &f, uint64_t k, uint64_t *w, bool &negative) {
    w[0] = f.constant_low;
    w[1] = f.constant_high;
    size_t n = f.constant_high ? 2 : 1;
    negative = f.negative != (f.alternating && (k & 1));
    for (unsigned i = 0; i < f.count && n; ++i) {
        const auto &factor = f.factor[i];
        const i128 value = factor_value(factor, k);
        const u128 magnitude = value < 0 ? u128(-(value + 1)) + 1 : u128(value);
        if (!magnitude)
            return 0;
        if (value < 0 && (factor.power & 1))
            negative = !negative;
        for (unsigned e = 0; e < factor.power; ++e) {
            if (magnitude >> 64)
                n = times2(w, n, uint64_t(magnitude), uint64_t(magnitude >> 64));
            else
                n = times(w, n, uint64_t(magnitude));
            require(n <= max_leaf_words, SBN3_FATAL_WORKSPACE, "formula leaf words");
        }
    }
    if (f.degree && n) {
        // Horner in 128 bits; preparation bounds |value| < 2^127 on the domain.
        i128 value = f.coefficient[f.degree];
        for (unsigned i = f.degree; i-- > 0;)
            value = value * i128(k) + f.coefficient[i];
        const u128 magnitude = value < 0 ? u128(-(value + 1)) + 1 : u128(value);
        if (!magnitude)
            return 0;
        if (value < 0)
            negative = !negative;
        n = magnitude >> 64 ? times2(w, n, uint64_t(magnitude), uint64_t(magnitude >> 64))
                            : times(w, n, uint64_t(magnitude));
        require(n <= max_leaf_words, SBN3_FATAL_WORKSPACE, "formula leaf words");
    }
    return n;
}
// log2 upper bound of |c * prod (a k + b)^e * poly(k)| at one index, using
// a k + |b| and sum |coefficients| k^degree.
long double log2_upper_abs(const FactorProduct &f, uint64_t k) {
    long double s = log2l((long double)f.constant_low + ldexpl((long double)f.constant_high, 64));
    for (unsigned i = 0; i < f.count; ++i) {
        const auto &x = f.factor[i];
        const long double m = x.b < 0 ? (long double)(uint64_t(-(x.b + 1)) + 1) : (long double)x.b;
        s += x.power * log2l((long double)x.a * (long double)k + m);
    }
    if (f.degree) {
        long double sum = 0;
        for (unsigned i = 0; i <= f.degree; ++i)
            sum += fabsl((long double)f.coefficient[i]);
        s += log2l(sum) + f.degree * log2l((long double)std::max<uint64_t>(k, 1));
    }
    return s;
}
// log2 of a sign-definite product c * prod (a k + b)^e at one index (exact
// values are both its lower and upper envelope), in long double.
long double log2_lower(const FactorProduct &f, uint64_t k) {
    long double s = log2l((long double)f.constant_low + ldexpl((long double)f.constant_high, 64));
    for (unsigned i = 0; i < f.count; ++i) {
        const auto &x = f.factor[i];
        s += x.power * log2l((long double)x.a * (long double)k + (long double)x.b);
    }
    return s;
}
// Antiderivative of log2(a x + c): ((a x + c) log2(a x + c) - (a x + c)/ln 2)/a.
long double antiderivative(long double a, long double c, long double x) {
    const long double v = a * x + c;
    return v <= 0 ? 0 : (v * log2l(v) - v / logl(2.0L)) / a;
}
// Rounding allowance in bits for a real bound of magnitude `mass`.
long double allowance(long double mass) {
    return 64 * std::numeric_limits<long double>::epsilon() * (fabsl(mass) + 1) + 2;
}
u128 hash_product(u128 h, const FactorProduct &f) {
    for (uint64_t x : {f.constant_low, f.constant_high, uint64_t(f.negative), uint64_t(f.alternating),
                       uint64_t(f.count)})
        h = identity::word(uint64_t(h), x);
    for (unsigned i = 0; i < f.count; ++i)
        for (uint64_t x : {f.factor[i].a, uint64_t(f.factor[i].b), uint64_t(f.factor[i].power)})
            h = identity::word(uint64_t(h), x);
    h = identity::word(uint64_t(h), f.degree);
    for (unsigned i = 0; i <= f.degree; ++i)
        h = identity::word(uint64_t(h), uint64_t(f.coefficient[i]));
    return h;
}
// Direction of f(x) = log2 Q(x) - log2 R(x) for normalized, positive linear
// factors (a >= 1, a x + b > 0 on the domain). With offsets u = b/a of Q's
// factors and v of R's (each repeated by its power),
//   f'(x) ln 2 = sum_i 1/(x + u_i) - sum_j 1/(x + v_j),
// and 1/(x + u) decreases in u. If every R factor can be paired with its own
// Q factor of smaller or equal offset, each pair and every unpaired Q factor
// contributes >= 0: f is nondecreasing (+1). The mirrored pairing gives -1.
// Sorted offsets decide the pairing; 0 when neither holds.
int log_ratio_direction(const FactorProduct &q, const FactorProduct &r) {
    struct Offset {
        uint64_t a;
        int64_t b;
    };
    Offset u[max_formula_factors * 16], v[max_formula_factors * 16];
    unsigned m = 0, n = 0;
    auto collect = [](const FactorProduct &f, Offset *out, unsigned &count) {
        for (unsigned i = 0; i < f.count; ++i)
            for (unsigned e = 0; e < f.factor[i].power; ++e)
                out[count++] = {f.factor[i].a, f.factor[i].b};
    };
    collect(q, u, m);
    collect(r, v, n);
    // b1/a1 < b2/a2 with positive a: compare b1*a2 with b2*a1 (|b| < 2^63, a < 2^64: below 2^127).
    auto less = [](const Offset &x, const Offset &y) { return i128(x.b) * i128(y.a) < i128(y.b) * i128(x.a); };
    std::sort(u, u + m, less);
    std::sort(v, v + n, less);
    auto dominated = [&](const Offset *small, unsigned small_count, const Offset *large, unsigned large_count) {
        // every element of `large` (count <= small_count) pairs with a distinct element of `small` not above it
        if (large_count > small_count)
            return false;
        for (unsigned j = 0; j < large_count; ++j)
            if (less(large[j], small[j]))
                return false;
        return true;
    };
    if (dominated(u, m, v, n))
        return 1;
    if (dominated(v, n, u, m))
        return -1;
    return 0;
}
// Exact-span fact: |prod_{k in [lo,hi)} X(k)| < 2^bits. The integer envelope
// sum adds a rounded-up bit length per factor and term (about half a bit too
// much each, 1-5% in all); for a sign-definite factor product the logarithmic
// estimator gives the span to within a bit. The exact-span fact needs that:
// a loose span makes an exact tree look truncated (e at 1024 limbs misses its
// own demand by 5% of integer slack). Capacities keep the integer sums: they
// are position-independent, so equal subtrees and blocks meet equal sizes and
// share plans, a planning pass asks for thousands of them (in query and again
// in bind), and estimator-sized capacities bought no execution time (A/B at
// 2^20 limbs: 0.98-1.02, products run at their actual operand sizes).
constexpr uint64_t span_floor_bits = 1024;
u128 product_span_bits(const FactorProduct &f, const ProductAnalysis &a, uint64_t lo, uint64_t hi) {
    const u128 integer = a.up_sum(lo, hi);
    if (integer < span_floor_bits || !a.sign_definite || lo >= hi)
        return integer;
    LogBounds real{};
    if (!product_log2_sum(f, lo, hi, real) || !(real.upper >= 0))
        return integer;
    const long double bits = floorl(real.upper) + 1;
    return bits < (long double)integer ? u128(bits) : integer;
}
} // namespace

void FactorBitTable::build(uint64_t a_, uint64_t c_, uint64_t begin_, uint64_t end_) noexcept {
    *this = {};
    a = a_;
    c = c_;
    begin = begin_;
    end = end_;
    if (begin >= end)
        return;
    uint64_t k = begin, total = 0;
    first_width = bits(u128(a) * k + c); // width 0 (value 0) is legal for an upper envelope
    for (;;) {
        const unsigned w = bits(u128(a) * k + c);
        // Widths can jump by more than one between consecutive indices when
        // a > 1; skipped widths get empty ranges starting at the same index.
        require(w >= first_width + count && w - first_width < 128, SBN3_FATAL_MATH, "factor width table");
        while (first_width + count <= w) {
            first[count] = k;
            cumulative[count] = total;
            ++count;
        }
        uint64_t next = end;
        if (a) {
            const u128 threshold = u128(1) << w; // first value with width w+1
            if (threshold > c) {
                const u128 kk = (threshold - c + a - 1) / a;
                if (kk < end)
                    next = uint64_t(kk);
            }
        }
        total += (next - k) * w;
        if (next == end)
            break;
        k = next;
    }
}
uint64_t FactorBitTable::prefix(uint64_t k) const noexcept {
    if (!count || k <= begin)
        return 0;
    require(k <= end, SBN3_FATAL_ARGUMENT, "factor table domain");
    unsigned lo = 0, hi = count;
    while (hi - lo > 1) {
        const unsigned m = lo + (hi - lo) / 2;
        if (first[m] <= k)
            lo = m;
        else
            hi = m;
    }
    return cumulative[lo] + (k - first[lo]) * (first_width + lo);
}
unsigned FactorBitTable::width(uint64_t k) const noexcept {
    require(count && k >= begin && k < end, SBN3_FATAL_ARGUMENT, "factor width domain");
    unsigned lo = 0, hi = count;
    while (hi - lo > 1) {
        const unsigned m = lo + (hi - lo) / 2;
        if (first[m] <= k)
            lo = m;
        else
            hi = m;
    }
    return first_width + lo;
}
uint64_t ProductAnalysis::up_sum(uint64_t lo_k, uint64_t hi_k) const noexcept {
    if (hi_k <= lo_k)
        return 0;
    uint64_t s = (hi_k - lo_k) * constant_up;
    for (unsigned i = 0; i < factors; ++i)
        s += uint64_t(up[i].sum(lo_k, hi_k)) * power[i];
    return s;
}
uint64_t ProductAnalysis::lo_sum(uint64_t lo_k, uint64_t hi_k) const noexcept {
    if (hi_k <= lo_k)
        return 0;
    require(has_lower, SBN3_FATAL_ARGUMENT, "product lower envelope");
    uint64_t s = (hi_k - lo_k) * (constant_lo - degree);
    for (unsigned i = 0; i < factors; ++i)
        s += uint64_t(lo[i].sum(lo_k, hi_k)) * power[i];
    return s;
}
unsigned ProductAnalysis::up_at(uint64_t k) const noexcept {
    unsigned s = constant_up;
    for (unsigned i = 0; i < factors; ++i)
        s += up[i].width(k) * power[i];
    return s;
}
unsigned ProductAnalysis::lo_at(uint64_t k) const noexcept {
    require(has_lower, SBN3_FATAL_ARGUMENT, "product lower envelope");
    unsigned s = constant_lo;
    for (unsigned i = 0; i < factors; ++i)
        s += lo[i].width(k) * power[i];
    return s - degree;
}
namespace {
// Fold constant factors (a == 0) into the constant and validate structure.
bool normalize_product(FactorProduct &f, const char *&why) {
    u128 c = constant(f);
    if (!c) {
        why = "zero constant";
        return false;
    }
    if (f.degree > max_polynomial_degree || (f.degree && !f.coefficient[f.degree])) {
        why = "polynomial degree outside 1..4 or zero leading coefficient";
        return false;
    }
    FactorProduct out = f;
    out.count = 0;
    for (unsigned i = 0; i < f.count; ++i) {
        const auto &x = f.factor[i];
        if (!x.power || x.power > 16) {
            why = "factor power outside 1..16";
            return false;
        }
        if (!x.a) {
            if (!x.b) {
                why = "zero constant factor";
                return false;
            }
            const u128 m = x.b < 0 ? u128(-(i128(x.b) + 1)) + 1 : u128(x.b);
            for (unsigned e = 0; e < x.power; ++e) {
                if (bits(c) + bits(m) > 128) {
                    why = "constant exceeds 128 bits";
                    return false;
                }
                c *= m;
            }
            if (x.b < 0 && (x.power & 1))
                out.negative = !out.negative;
            continue;
        }
        out.factor[out.count++] = x;
    }
    // Preparation-time known powers of two: a factor with both coefficients
    // even carries 2^s per term; move it into the constant so the 2-adic
    // valuation is visible to the whole-word normalization bound.
    for (unsigned i = 0; i < out.count; ++i) {
        auto &x = out.factor[i];
        if (!x.b) {
            // a*k: the whole coefficient is a per-term constant factor.
            if (x.a == 1)
                continue;
            u128 scaled = c;
            bool fits = true;
            for (unsigned e = 0; e < x.power && fits; ++e) {
                fits = bits(scaled) + bits(x.a) <= 128;
                if (fits)
                    scaled *= x.a;
            }
            if (fits) {
                c = scaled;
                x.a = 1;
            }
            continue;
        }
        const uint64_t magnitude = x.b < 0 ? uint64_t(-(x.b + 1)) + 1 : uint64_t(x.b);
        const unsigned s = std::min(unsigned(__builtin_ctzll(x.a)), unsigned(__builtin_ctzll(magnitude)));
        if (!s)
            continue;
        if (bits(c) + s * x.power > 128)
            continue;
        x.a >>= s;
        x.b = x.b < 0 ? -int64_t((magnitude >> s)) : int64_t(magnitude >> s);
        c <<= s * x.power;
    }
    out.constant_low = uint64_t(c);
    out.constant_high = uint64_t(c >> 64);
    f = out;
    return true;
}
bool analyze_product(const FactorProduct &input, uint64_t begin, uint64_t end, bool need_lower, ProductAnalysis &a,
                     const char *&why) {
    a = {};
    FactorProduct f = input;
    if (f.degree) {
        if (need_lower) {
            why = "polynomial multipliers are supported in numerators only";
            return false;
        }
        // |sum c_i k^i| <= (sum |c_i|) (k+1)^degree: fold into the upper envelope.
        u128 sum = 0;
        for (unsigned i = 0; i <= f.degree; ++i)
            sum += f.coefficient[i] < 0 ? u128(-(f.coefficient[i] + 1)) + 1 : u128(f.coefficient[i]);
        if (f.count >= max_formula_factors || bits(constant(f)) + bits(sum) > 128) {
            why = "polynomial envelope does not fit";
            return false;
        }
        // Horner domain: |value(k)| < 2^127 for every k below end (largest index end-1).
        if (end && bits(sum) + f.degree * bits(u128(end - 1)) > 126) {
            why = "polynomial value exceeds 127 bits on this domain";
            return false;
        }
        const u128 scaled = constant(f) * sum;
        f.constant_low = uint64_t(scaled);
        f.constant_high = uint64_t(scaled >> 64);
        f.factor[f.count++] = {1, 1, f.degree};
        f.degree = 0;
    }
    const u128 c = constant(f);
    a.constant_up = bits(c) - (power_of_two(c) ? 1 : 0);
    a.constant_lo = bits(c);
    a.twos = unsigned(c ? (uint64_t(c) ? __builtin_ctzll(uint64_t(c)) : 64 + __builtin_ctzll(uint64_t(c >> 64))) : 0);
    a.factors = f.count;
    a.sign_definite = true;
    for (unsigned i = 0; i < f.count; ++i) {
        const auto &x = f.factor[i];
        a.power[i] = x.power;
        a.degree += x.power;
        const uint64_t absolute = uint64_t(x.b < 0 ? -(x.b + 1) : x.b) + (x.b < 0 ? 1 : 0);
        if (begin < end)
            a.up[i].build(x.a, absolute, begin, end);
        // Positive on the whole domain iff positive at its first index.
        const bool positive = begin >= end || factor_value(x, begin) >= 1;
        a.sign_definite = a.sign_definite && positive;
    }
    if (need_lower) {
        if (!a.sign_definite) {
            why = "denominator/ratio factor is zero or changes sign on the domain";
            return false;
        }
        for (unsigned i = 0; i < f.count; ++i) {
            const auto &x = f.factor[i];
            if (x.b >= 0)
                a.lo[i] = a.up[i];
            else if (begin < end) {
                // Sign-definite: a k + b >= 1 on the domain, so the true value
                // is also the tightest monotone upper envelope.
                // a*k+b >= 1 with b < 0: shift the origin so the form is a*k' + c' with c' >= 0.
                // Smallest k0 with a*k0 + b >= 0, computed in 128 bits: |b| + a - 1
                // overflows 64 bits for wide coefficients. c0 = a*k0 + b lies in [0, a).
                const u128 magnitude = u128(-(x.b + 1)) + 1;
                const u128 k0_wide = (magnitude + x.a - 1) / x.a;
                require(k0_wide <= uint64_t(1) << 63, SBN3_FATAL_MATH, "factor lower envelope origin");
                const uint64_t k0 = uint64_t(k0_wide);
                const i128 c0_wide = i128(u128(x.a) * k0) + x.b;
                require(c0_wide >= 0 && c0_wide < i128(x.a) && begin >= k0, SBN3_FATAL_MATH,
                        "factor lower envelope origin");
                const int64_t c0 = int64_t(c0_wide);
                a.lo[i].build(x.a, uint64_t(c0), begin - k0, end - k0);
                a.lo[i].begin = begin;
                a.lo[i].end = end;
                for (unsigned j = 0; j < a.lo[i].count; ++j)
                    a.lo[i].first[j] += k0;
                a.up[i] = a.lo[i];
            }
        }
        a.has_lower = true;
    }
    return true;
}
} // namespace

uint64_t formula_domain_limit(const FormulaDef &def) noexcept {
    uint64_t limit = max_index + 1;
    for (const auto *f : {&def.P, &def.Q, &def.R}) {
        if (!f->degree || f->degree > max_polynomial_degree)
            continue;
        u128 sum = 0;
        for (unsigned i = 0; i <= f->degree; ++i)
            sum += f->coefficient[i] < 0 ? u128(-(f->coefficient[i] + 1)) + 1 : u128(f->coefficient[i]);
        // analyze_product requires bits(sum) + degree * bits(end) <= 126.
        const unsigned available = bits(sum) >= 126 ? 0 : (126 - bits(sum)) / f->degree;
        const uint64_t end = available >= 48 ? max_index + 1 : uint64_t(1) << available; // bits(end) <= available
        limit = std::min(limit, end);
    }
    return limit;
}
bool data_formula_prepare(const FormulaDef &input, uint64_t index_end, DataFormula &out, bool probe_only) noexcept {
    out = {};
    out.probe_only = probe_only;
    out.def = input;
    out.index_end = index_end;
    auto reject = [&](const char *why) {
        out.rejection = why;
        return false;
    };
    auto &def = out.def;
    if (def.recipe < SBN3_SERIES_HYPERDESCENT || def.recipe > SBN3_SERIES_BINARY_BBP)
        return reject("unknown recipe");
    if (def.begin >= index_end || index_end - 1 > max_index)
        return reject("domain must be nonempty with indices below 2^48");
    const bool common = def.recipe == SBN3_SERIES_COMMON_P2B3, bbp = def.recipe == SBN3_SERIES_BINARY_BBP;
    const bool hyper = def.recipe == SBN3_SERIES_HYPERDESCENT;
    if (bbp) {
        if (!def.stride)
            return reject("BinaryBBP stride must be positive");
        if (u128(def.stride) * (index_end - 1) >= (u128(1) << 62) || def.shift >= (int64_t(1) << 62) ||
            def.shift <= -(int64_t(1) << 62))
            return reject("BinaryBBP exponent range");
    } else if (def.stride || def.shift)
        return reject("shift/stride are BinaryBBP-only");
    const char *why = nullptr;
    for (auto *f : {&def.P, &def.Q, &def.R})
        if (!normalize_product(*f, why))
            return reject(why);
    if (!common) {
        const bool unit = def.R.count == 0 && constant(def.R) == 1 && !def.R.negative && !def.R.alternating;
        if (!unit)
            return reject(hyper ? "Hyperdescent requires R == 1" : "BinaryBBP has no R");
    }
    if (hyper && (def.Q.negative || def.Q.alternating))
        return reject("Hyperdescent requires Q > 0 (U == +1 cannot absorb a sign)");
    out.first_positive = def.begin + (def.explicit_first ? 1 : 0);
    if (def.explicit_first) {
        if (!def.first.d)
            return reject("explicit first term has zero denominator");
        if (common && !def.first.u)
            return reject("explicit first term has zero U");
        if (hyper && def.first.u != 1)
            return reject("Hyperdescent explicit first term must have U == 1");
        out.first_t_bits = bits(def.first.t);
        out.first_d_bits = bits(def.first.d);
        out.first_u_bits = common ? bits(def.first.u) : 0;
    }
    const uint64_t fp = out.first_positive;
    if (!analyze_product(def.P, fp, index_end, false, out.P, why) ||
        !analyze_product(def.Q, fp, index_end, true, out.Q, why) ||
        !analyze_product(def.R, fp, index_end, common, out.R, why))
        return reject(why);
    // Single-term leaf widths over the domain (monotone envelopes: last index).
    unsigned leaf_bits = std::max({out.first_t_bits, out.first_d_bits, out.first_u_bits, 1u});
    if (fp < index_end) {
        const uint64_t last = index_end - 1;
        leaf_bits = std::max({leaf_bits, out.P.up_at(last), out.Q.up_at(last), common ? out.R.up_at(last) : 0u});
    }
    out.leaf_words = unsigned(words(leaf_bits));
    if (out.leaf_words > max_leaf_words)
        return reject("single-term leaf exceeds 64 words on this domain");
    // Contraction certificate |R(k)| <= |Q(k)| and per-term attenuation. The
    // difference of two nondecreasing step functions attains its minimum at
    // the start or right after a jump of the subtrahend.
    int ratio = INT32_MIN;
    if (fp < index_end) {
        auto attenuation_at = [&](uint64_t k) {
            return int(out.Q.lo_at(k)) - int(common ? out.R.up_at(k) : 0) - 1;
        };
        int minimum = attenuation_at(fp);
        if (common)
            for (unsigned i = 0; i < out.R.factors; ++i)
                for (unsigned j = 0; j < out.R.up[i].count; ++j)
                    minimum = std::min(minimum, attenuation_at(out.R.up[i].first[j]));
        auto ratio_at = [&](uint64_t k) { return int(out.P.up_at(k)) - int(out.Q.lo_at(k)) + 1; };
        ratio = ratio_at(fp);
        for (unsigned i = 0; i < out.P.factors; ++i)
            for (unsigned j = 0; j < out.P.up[i].count; ++j)
                ratio = std::max(ratio, ratio_at(out.P.up[i].first[j]));
        ratio = std::max(ratio, ratio_at(index_end - 1));
        out.integer_contraction = minimum >= 0;
        out.attenuation_minimum = minimum >= 0 ? uint64_t(minimum) : 0;
    } else {
        out.integer_contraction = true;
        out.attenuation_minimum = 0;
    }
    out.contraction = out.integer_contraction;
    if (def.explicit_first) {
        // Exact small values: floor(log2 d) - ceil(log2 u) per-term attenuation.
        const unsigned u_bits = bits(def.first.u); // U == 1 for Hyperdescent, unused by BinaryBBP
        const int att_first = int(out.first_d_bits) - 1 - int(power_of_two(def.first.u) ? u_bits - 1 : u_bits);
        if (!bbp) {
            // The explicit term occurs at most once; attenuation_minimum stays
            // the factored-term minimum and the envelope adds this term separately.
            out.contraction = out.contraction && att_first >= 0;
            out.integer_contraction = out.integer_contraction && att_first >= 0;
            out.first_attenuation = att_first >= 0 ? uint64_t(att_first) : 0;
        }
        ratio = std::max(ratio, int(out.first_t_bits) - int(out.first_d_bits) + 1);
    }
    if (bbp) {
        out.contraction = true; // additive form: the exponent alone certifies attenuation
        out.attenuation_minimum = def.stride;
    }
    // Bucketed real-log certificate for the contribution side. Q is bounded
    // below at the bucket start and R above at the bucket end, so each
    // bucket floor is a pointwise lower bound for all of its indices. It can
    // certify contraction where the integer bit-length sums cannot.
    if (!bbp && fp < index_end && !probe_only) {
        auto &t = out.attenuation;
        t.begin = fp;
        t.end = index_end;
        bool bucket_contraction = true;
        uint64_t k = fp;
        u128 total = 0; // 2^-fraction_bits bit
        const long double unit = (long double)(1u << AttenuationTable::fraction_bits);
        out.ratio_direction = log_ratio_direction(def.Q, def.R);
        while (k < index_end) {
            uint64_t next = std::max(k + 1, k + k / 32);
            // Never straddle a power of two: factor widths jump there and a
            // bucket floor taken at its start would be paid by every index.
            const uint64_t boundary = k ? uint64_t(1) << bits(k) : 1;
            if (boundary > k && next > boundary)
                next = boundary;
            if (next > index_end)
                next = index_end;
            // Real floor with a tiny libm allowance, or the integer bit-length
            // certificate on the same bucket, whichever is stronger; both are
            // pointwise lower bounds for every index of the bucket. A proven
            // monotone ratio attains its bucket minimum at one end; otherwise
            // Q (increasing) is taken at the start and R (increasing) at the end.
            const uint64_t q_at = out.ratio_direction < 0 ? next - 1 : k;
            const uint64_t r_at = out.ratio_direction > 0 ? k : next - 1;
            const long double lo_q = log2_lower(def.Q, q_at);
            const long double hi_r = common ? log2_lower(def.R, r_at) : 0; // R is sign-definite: exact values
            const long double real_floor =
                floorl((lo_q - hi_r - (fabsl(lo_q) + fabsl(hi_r)) * 1e-15L - 1e-9L) * unit);
            const int integer_floor = int(out.Q.lo_at(k)) - int(common ? out.R.up_at(next - 1) : 0) - 1;
            const long double floor_units = std::max(real_floor, (long double)integer_floor * unit);
            if (floor_units < 0) {
                bucket_contraction = false;
                break;
            }
            require(t.count < 2048, SBN3_FATAL_MATH, "attenuation table capacity");
            t.first[t.count] = k;
            t.cumulative[t.count] = uint64_t(std::min<u128>(total >> AttenuationTable::fraction_bits, u128(INT64_MAX)));
            t.fraction[t.count] = uint8_t(total & ((1u << AttenuationTable::fraction_bits) - 1));
            t.per_term[t.count] = uint32_t(std::min(floor_units, 4.0e9L));
            ++t.count;
            total += u128(next - k) * t.per_term[t.count - 1];
            k = next;
        }
        if (!bucket_contraction)
            t = {};
        out.contraction = bucket_contraction || out.integer_contraction;
    }
    out.ratio_bits = ratio > 0 ? unsigned(ratio) : 0;
    out.smooth_split = common && out.contraction;
    if (def.extract_twos) {
        if (hyper)
            return reject("Hyperdescent cannot rescale its implicit U");
        out.twos_shift = out.Q.twos;
    }
    // Batch leaf: single-word Q on the domain and a constant single-word P.
    // Batch numerator per term: constant, or constant * polynomial(k) with no
    // factors, evaluated in 128 bits and required to fit a signed word on the domain.
    bool p_word = def.P.count == 0 && !def.P.constant_high && def.P.constant_low < (uint64_t(1) << 63);
    if (p_word && def.P.degree) {
        u128 sum = 0;
        for (unsigned i = 0; i <= def.P.degree; ++i)
            sum += def.P.coefficient[i] < 0 ? u128(-(def.P.coefficient[i] + 1)) + 1 : u128(def.P.coefficient[i]);
        p_word = bits(u128(def.P.constant_low) * sum) + def.P.degree * bits(u128(index_end)) <= 62;
    }
    const bool q_word = fp >= index_end || out.Q.up_at(index_end - 1) <= 64;
    const bool first_word = !def.explicit_first || def.first.t < (uint64_t(1) << 63);
    out.leaf_kind = hyper && p_word && q_word && first_word ? LeafKind::HyperWordBatch : LeafKind::Factored;
    if (common && !out.twos_shift && q_word &&
        (fp >= index_end || (out.P.up_at(index_end - 1) <= 64 && out.R.up_at(index_end - 1) <= 64)))
        out.leaf_kind = LeafKind::CommonWordBatch;
    // Wide batch: every product is (constant, at most 128 bits) times an index
    // part that stays below 2^126 on the domain, so a term is evaluated in
    // 128-bit words and applied with one- or two-limb passes. Growth per term
    // is at most three words.
    if (common && !out.twos_shift && out.leaf_kind == LeafKind::Factored && fp < index_end) {
        auto index_bits = [&](const FactorProduct &f) {
            unsigned total = 0;
            for (unsigned i = 0; i < f.count; ++i) {
                const auto &x = f.factor[i];
                const u128 magnitude = x.b < 0 ? u128(-(x.b + 1)) + 1 : u128(x.b);
                total += x.power * bits(u128(x.a) * (index_end - 1) + magnitude);
            }
            if (f.degree) {
                u128 sum = 0;
                for (unsigned i = 0; i <= f.degree; ++i)
                    sum += f.coefficient[i] < 0 ? u128(-(f.coefficient[i] + 1)) + 1 : u128(f.coefficient[i]);
                total += bits(sum) + f.degree * bits(u128(index_end - 1));
            }
            return total;
        };
        const unsigned worst =
            std::max({out.P.up_at(index_end - 1), out.Q.up_at(index_end - 1), out.R.up_at(index_end - 1)});
        if (index_bits(out.def.P) <= 126 && index_bits(out.def.Q) <= 126 && index_bits(out.def.R) <= 126 &&
            worst <= 192 && first_word) {
            out.leaf_kind = LeafKind::CommonWideBatch;
            out.batch_words = std::max(1u, (worst + 63) / 64);
        }
    }
    u128 h = identity::word(identity::fnv_seed, 0x444154414f524d31ULL);
    for (uint64_t x : {uint64_t(def.recipe), def.begin, uint64_t(def.shift), uint64_t(def.stride),
                       uint64_t(def.explicit_first), def.first.t, def.first.d, def.first.u,
                       uint64_t(def.first.t_negative), uint64_t(def.extract_twos)})
        h = identity::word(uint64_t(h), x);
    for (const auto *f : {&def.P, &def.Q, &def.R})
        h = hash_product(h, *f);
    out.formula_id = uint64_t(h);
    out.parameter_id = identity::word(identity::word(identity::fnv_seed, index_end), uint64_t(out.leaf_kind));
    // Structural convergence: degrees and leading behaviour of the factored
    // products decide whether the infinite tail can be certified at all.
    {
        auto degree_of = [](const FactorProduct &f) {
            unsigned d = f.degree;
            for (unsigned i = 0; i < f.count; ++i)
                d += f.factor[i].power;
            return d;
        };
        const unsigned deg_q = degree_of(def.Q), deg_r = common ? degree_of(def.R) : 0;
        if (bbp)
            out.tail_rejection = nullptr; // 2^-stride per term is a geometric factor by itself
        else if (deg_r > deg_q)
            out.tail_rejection = "term ratio |R(k)/Q(k)| grows without bound: the infinite series diverges";
        else if (deg_r == deg_q) {
            // Leading coefficients decide the limit ratio.
            long double lead_r = log2l((long double)def.R.constant_low + ldexpl((long double)def.R.constant_high, 64));
            long double lead_q = log2l((long double)def.Q.constant_low + ldexpl((long double)def.Q.constant_high, 64));
            for (unsigned i = 0; i < def.R.count; ++i)
                lead_r += def.R.factor[i].power * log2l((long double)def.R.factor[i].a);
            for (unsigned i = 0; i < def.Q.count; ++i)
                lead_q += def.Q.factor[i].power * log2l((long double)def.Q.factor[i].a);
            if (!(lead_r < lead_q - 1e-12L))
                out.tail_rejection = "term ratio |R(k)/Q(k)| does not tend below 1: no geometric tail bound";
        }
    }
    return true;
}
namespace {
struct Window {
    u128 t = 0, d = 0, u = 0;
};
} // namespace
sbn3_query_result DataFormula::bounds(sbn3_series_range r, uint64_t n, unsigned need, sbn3_series_shape &shape,
                                      bool normalized, bool span_only) const noexcept {
    shape = {};
    const bool common = def.recipe == SBN3_SERIES_COMMON_P2B3, bbp = def.recipe == SBN3_SERIES_BINARY_BBP;
    const unsigned allowed = common ? 7u : 3u;
    if (rejection || r.begin < def.begin || r.end > index_end || r.begin >= r.end || !n ||
        n > r.end - r.begin || !need || (need & ~allowed))
        return SBN3_UNSUPPORTED;
    Window best{};
    auto window = [&](uint64_t w0, uint64_t w1) {
        const bool explicit_in = def.explicit_first && w0 == def.begin;
        const uint64_t f0 = std::max(w0, first_positive), f1 = std::max(f0, w1), m = f1 - f0, count = w1 - w0;
        u128 d = (span_only ? product_span_bits(def.Q, Q, f0, f1) : u128(Q.up_sum(f0, f1))) + (explicit_in ? first_d_bits : 0);
        const u128 d_stored = d - u128(twos_shift) * m; // Q(k) >= 2^twos, so this never underflows
        const u128 u = common ? (span_only ? product_span_bits(def.R, R, f0, f1) : u128(R.up_sum(f0, f1))) +
                                    (explicit_in ? first_u_bits : 0)
                              : 0;
        const unsigned p_max = std::max(m ? P.up_at(f1 - 1) : 0u, explicit_in ? first_t_bits : 0u);
        u128 t;
        if (bbp)
            t = d + ceil_log(count) + p_max + u128(def.stride) * (count - 1) + 1;
        else if (contraction) {
            // |T/D| <= sum_k |P(k)/Q(k)| when every |R(j)| <= |Q(j)|.
            int ratio = m ? int(P.up_at(f1 - 1)) - int(Q.lo_at(f0)) + 1 : INT32_MIN;
            if (explicit_in)
                ratio = std::max(ratio, int(first_t_bits) - int(first_d_bits) + 1);
            t = d + ceil_log(count) + (ratio > 0 ? unsigned(ratio) : 0u) + 1;
        } else
            t = ceil_log(count) + p_max + u + d + 1;
        d = d_stored + 1;
        // Every factored Q term carries 2^twos; canonical whole-word mantissas
        // omit those low zero words except for at most one partial word.
        if (normalized && Q.twos && !twos_shift && m)
            d = std::min(d, d - u128(Q.twos) * m + 63);
        best.t = std::max(best.t, t);
        best.d = std::max(best.d, d);
        best.u = std::max(best.u, u + 1);
    };
    // Envelopes grow with the index, so the last n terms dominate every other
    // n-term subrange, except when an explicit first term can be heavier.
    window(r.end - n, r.end);
    if (def.explicit_first && r.begin == def.begin && r.end - n > r.begin)
        window(r.begin, r.begin + n);
    const u128 limit = u128(SIZE_MAX / 8) * 64;
    if (std::max({best.t, best.d, best.u}) > limit)
        return SBN3_QUERY_CAPACITY;
    const u128 sizes[]{best.t, best.d, best.u};
    for (unsigned j = 0; j < 3; ++j)
        if (need & (1u << j))
            shape.limbs[j] = std::max(size_t(1), words(sizes[j]));
    return SBN3_SUPPORTED;
}
uint64_t DataFormula::mass(sbn3_series_range r) const noexcept {
    // Per-term measure up_Q(k)+1 keeps every term positive so equal-measure
    // cuts bound recursion depth even for constant denominators.
    const bool explicit_in = def.explicit_first && r.begin == def.begin;
    const uint64_t f0 = std::max(r.begin, first_positive), f1 = std::max(f0, r.end);
    return Q.up_sum(f0, f1) + (f1 - f0) + (explicit_in ? first_d_bits + 1 : 0);
}
double DataFormula::work(sbn3_series_range r) const noexcept {
    return double(mass(r));
}
uint64_t DataFormula::split_point(sbn3_series_range r, double fraction) const noexcept {
    const uint64_t length = r.end - r.begin;
    require(length >= 2 && fraction > 0 && fraction < 1, SBN3_FATAL_ARGUMENT, "formula split fraction");
    const double target = double(mass(r)) * fraction;
    uint64_t lo = r.begin + 1, hi = r.end - 1;
    while (lo < hi) {
        const uint64_t m = lo + (hi - lo) / 2;
        if (double(mass({r.begin, m})) < target)
            lo = m + 1;
        else
            hi = m;
    }
    if (lo > r.begin + 1 && std::abs(double(mass({r.begin, lo - 1})) - target) <
                                std::abs(double(mass({r.begin, lo})) - target))
        --lo;
    return lo;
}
sbn3_query_result DataFormula::serial_envelope(sbn3_series_range r, unsigned depth, unsigned need,
                                               uint64_t &max_terms, sbn3_series_shape &shape,
                                               bool normalized) const noexcept {
    shape = {};
    if (!smooth_split || rejection || r.begin < def.begin || r.end > index_end || r.begin >= r.end || !need ||
        (need & ~7u) || depth > 96)
        return SBN3_UNSUPPORTED;
    const bool explicit_in = def.explicit_first && r.begin == def.begin;
    const uint64_t f0 = std::max(r.begin, first_positive), m = r.end > f0 ? r.end - f0 : 0;
    // Measure M = sum (up_Q + 1) over the range; each term contributes at
    // least low and at most high to it.
    uint64_t M = mass(r);
    const unsigned high = std::max(m ? Q.up_at(r.end - 1) + 1 : 0u, explicit_in ? first_d_bits + 1 : 0u);
    // Every factored term contributes at least low to the measure; the
    // explicit first term occurs at most once in any descendant.
    const unsigned low = std::max(1u, m ? Q.lo_at(f0) + 1 : 1u);
    const unsigned extra_term = explicit_in ? 1 : 0;
    int ratio = m ? int(P.up_at(r.end - 1)) - int(Q.lo_at(f0)) + 1 : INT32_MIN;
    if (explicit_in)
        ratio = std::max(ratio, int(first_t_bits) - int(first_d_bits) + 1);
    const unsigned ratio_bits_window = ratio > 0 ? unsigned(ratio) : 0;
    max_terms = r.end - r.begin;
    for (unsigned j = 0; j < depth; ++j) {
        // An equal-measure cut leaves each side at most half the parent plus
        // half of the cut term; both sides keep at least one term.
        M = (M + high + 1) / 2 + 1;
        const uint64_t fewer = max_terms > 1 ? max_terms - 1 : 1;
        max_terms = std::max<uint64_t>(1, std::min(fewer, M / low + extra_term));
    }
    // sum up_Q over factored terms <= M and <= (high-1)*terms; the explicit
    // term adds its own exact widths.
    const u128 raw_db = u128(M) + 1;
    u128 db = raw_db, ub = u128(explicit_in ? first_u_bits : 0) + 1;
    if (high > 1) {
        const unsigned span = high - 1;
        if (normalized && Q.twos)
            db = std::min(db, (u128(M) * (span - std::min(span, Q.twos)) + span - 1) / span + 64 +
                                  (explicit_in ? first_d_bits : 0));
        if (integer_contraction) {
            const unsigned c = std::min<unsigned>(span, unsigned(std::min<uint64_t>(attenuation_minimum + 1, span)));
            ub += (u128(M) * (span - c) + span - 1) / span;
        } else {
            // No per-term bit relation between R and Q: bound U by the heaviest
            // window of max_terms factored terms directly (R envelopes increase).
            const uint64_t heavy = std::max<uint64_t>(f0, r.end > max_terms ? r.end - max_terms : f0);
            ub += R.up_sum(heavy, r.end);
        }
    }
    const u128 tb = raw_db + ceil_log(max_terms) + ratio_bits_window + 1;
    const u128 sizes[]{tb, db, ub};
    for (unsigned j = 0; j < 3; ++j)
        if (need & (1u << j))
            shape.limbs[j] = std::max(size_t(1), words(sizes[j]));
    return SBN3_SUPPORTED;
}
uint64_t AttenuationTable::prefix(uint64_t k) const noexcept {
    if (!count || k <= begin)
        return 0;
    require(k <= end, SBN3_FATAL_ARGUMENT, "attenuation table domain");
    unsigned lo = 0, hi = count;
    while (hi - lo > 1) {
        const unsigned m = lo + (hi - lo) / 2;
        if (first[m] <= k)
            lo = m;
        else
            hi = m;
    }
    const u128 inside = (u128(k - first[lo]) * per_term[lo] + fraction[lo]) >> fraction_bits;
    return uint64_t(std::min<u128>(u128(cumulative[lo]) + inside, u128(INT64_MAX)));
}
uint64_t DataFormula::attenuation_bits(uint64_t a) const noexcept {
    require(a >= def.begin && a <= index_end, SBN3_FATAL_ARGUMENT, "formula attenuation index");
    if (a == def.begin)
        return 0;
    if (def.recipe == SBN3_SERIES_BINARY_BBP)
        return uint64_t(def.stride) * (a - def.begin);
    if (!contraction)
        return 0;
    uint64_t s = def.explicit_first ? first_attenuation : 0;
    if (a > first_positive) {
        // Prefer the bucketed real-log certificate; fall back to the integer
        // bit-length sums when no table was built.
        if (attenuation.count)
            s += attenuation.prefix(a);
        else if (integer_contraction)
            s += Q.lo_sum(first_positive, a) - R.up_sum(first_positive, a) - (a - first_positive);
    }
    return s;
}
long double DataFormula::attenuation_total(uint64_t a) const noexcept {
    require(a >= def.begin && a <= index_end, SBN3_FATAL_ARGUMENT, "formula attenuation index");
    // The cumulative integral bound below is a lower bound of the sum of log2|Q/R| whether or not
    // every single ratio is below one, so a probe may use it before contraction is decided.
    if (a == def.begin || def.recipe == SBN3_SERIES_BINARY_BBP || !(contraction || probe_only))
        return (long double)attenuation_bits(a);
    const bool common = def.recipe == SBN3_SERIES_COMMON_P2B3;
    long double s = def.explicit_first ? (long double)first_attenuation : 0;
    const uint64_t fp = first_positive;
    if (a <= fp)
        return s;
    // The logarithmic estimator has no integral-versus-sum slack (it is tight
    // to a fraction of a bit), so the stopping index it yields is minimal. It
    // applies to every accepted Q and R (positive linear factors); the integral
    // bound below remains as the fallback.
    {
        LogBounds q{}, r{};
        if (product_log2_sum(def.Q, fp, a, q) && (!common || product_log2_sum(def.R, fp, a, r)))
            return std::max(s + q.lower - r.upper, (long double)attenuation_bits(a));
    }
    // Combine identical factors before taking bounds. In particular a fixed
    // Q/R ratio has no spurious endpoint error from two cancelling integrals.
    struct SignedFactor {
        uint64_t a;
        int64_t b;
        int power;
    } factors[2 * max_formula_factors]{};
    unsigned count = 0;
    auto insert = [&](const FactorProduct &p, int sign) {
        for (unsigned i = 0; i < p.count; ++i) {
            const auto &f = p.factor[i];
            unsigned j = 0;
            for (; j < count; ++j)
                if (factors[j].a == f.a && factors[j].b == f.b)
                    break;
            if (j == count)
                factors[count++] = {f.a, f.b, 0};
            factors[j].power += sign * int(f.power);
        }
    };
    insert(def.Q, 1);
    if (common)
        insert(def.R, -1);
    auto constant_log = [](const FactorProduct &p) {
        return log2l((long double)p.constant_low + ldexpl((long double)p.constant_high, 64));
    };
    const long double qlog = constant_log(def.Q), rlog = common ? constant_log(def.R) : 0;
    long double error_mass = fabsl(s) + (a - fp) * (fabsl(qlog) + fabsl(rlog));
    s += (a - fp) * (qlog - rlog);
    for (unsigned j = 0; j < count; ++j) {
        const auto &f = factors[j];
        if (!f.power)
            continue;
        const long double slope = f.a, offset = f.b;
        long double value = log2l(slope * fp + offset);
        long double mass = fabsl(value);
        if (a > fp + 1) {
            // Positive factors take the lower integral; negative factors
            // subtract an upper integral, so the combined value stays low.
            const long double lo = f.power > 0 ? fp : fp + 1, hi = f.power > 0 ? a - 1 : a;
            const long double upper = antiderivative(slope, offset, hi),
                              lower = antiderivative(slope, offset, lo);
            value += upper - lower;
            mass += fabsl(upper) + fabsl(lower);
        }
        s += f.power * value;
        error_mass += std::abs(f.power) * mass;
    }
    // All are lower bounds of the exact total; the integer certificate can
    // be the stronger one for very short prefixes.
    // Account for the evaluated terms before subtraction, not only the
    // possibly much smaller result after cancellation.
    return std::max(s - allowance(error_mass), (long double)attenuation_bits(a));
}
long double DataFormula::value_log2_bound() const noexcept {
    require(!rejection && contraction, SBN3_FATAL_ARGUMENT, "formula value bound needs contraction");
    const bool common = def.recipe == SBN3_SERIES_COMMON_P2B3, bbp = def.recipe == SBN3_SERIES_BINARY_BBP;
    long double sum = def.explicit_first ? (long double)def.first.t / (long double)def.first.d : 0;
    const long double first_scale = def.explicit_first ? exp2l(-(long double)first_attenuation) : 1;
    if (bbp) {
        // |P(k)/Q(k)| 2^(shift - stride k): geometric in k with ratio 2^-stride.
        for (uint64_t k = first_positive; k < index_end && k < first_positive + 64; ++k)
            sum += exp2l(log2_upper_abs(def.P, k) - log2_lower(def.Q, k) + (long double)def.shift -
                         (long double)def.stride * (long double)k);
        // Beyond 64 terms the remaining geometric tail is below one more term.
        if (index_end > first_positive + 64)
            sum += exp2l(log2_upper_abs(def.P, first_positive + 64) - log2_lower(def.Q, first_positive + 64) +
                         (long double)def.shift - (long double)def.stride * (long double)(first_positive + 64) + 1);
        return log2l(sum) + 1e-9L;
    }
    if (!attenuation.count) {
        // Integer certificate only: |P/Q| < 2^ratio_bits, at most (count) terms with no attenuation.
        return (long double)ratio_bits + log2l((long double)(index_end - def.begin) + 1) + 1e-9L;
    }
    (void)common;
    for (unsigned b = 0; b < attenuation.count; ++b) {
        const uint64_t k0 = attenuation.first[b];
        const uint64_t k1 = b + 1 < attenuation.count ? attenuation.first[b + 1] : attenuation.end;
        if (k1 <= k0)
            continue;
        // |P(k)/Q(k)| is largest at the bucket end for P, smallest Q at its start.
        const long double pq = log2_upper_abs(def.P, k1 - 1) - log2_lower(def.Q, k0);
        const long double start = exp2l(-(long double)attenuation.prefix(k0)) * first_scale;
        const long double per =
            (long double)attenuation.per_term[b] / (long double)(1u << AttenuationTable::fraction_bits);
        // sum_{i<count} 2^(-per i): the geometric limit, never more than the number of terms.
        const long double terms = (long double)(k1 - k0);
        const long double geometric = per > 0 ? std::min(terms, 1 / (1 - exp2l(-per))) : terms;
        sum += exp2l(pq) * start * geometric;
    }
    return log2l(sum) + 1e-9L;
}
long double DataFormula::infinite_tail_bits(uint64_t N) const noexcept {
    require(!tail_rejection && N >= first_positive && N <= index_end && N >= 1, SBN3_FATAL_ARGUMENT,
            "formula infinite tail index");
    const bool common = def.recipe == SBN3_SERIES_COMMON_P2B3, bbp = def.recipe == SBN3_SERIES_BINARY_BBP;
    const long double n = (long double)N;
    auto constant_log2 = [](const FactorProduct &f) {
        return log2l((long double)f.constant_low + ldexpl((long double)f.constant_high, 64));
    };
    // Upper envelope for j >= N: c * prod (a + |b|/N)^e * j^deg (times sum|coef| j^degree for a polynomial).
    auto upper = [&](const FactorProduct &f, long double &log2_a, unsigned &deg) {
        log2_a = constant_log2(f);
        deg = f.degree;
        for (unsigned i = 0; i < f.count; ++i) {
            const auto &x = f.factor[i];
            const long double m = x.b < 0 ? (long double)(uint64_t(-(x.b + 1)) + 1) : (long double)x.b;
            log2_a += x.power * log2l((long double)x.a + m / n);
            deg += x.power;
        }
        if (f.degree) {
            long double sum = 0;
            for (unsigned i = 0; i <= f.degree; ++i)
                sum += fabsl((long double)f.coefficient[i]);
            log2_a += log2l(sum);
        }
    };
    // Lower envelope for j >= N (sign-definite, no polynomial): c * prod (a - |b|/N)^e * j^deg.
    auto lower = [&](const FactorProduct &f, long double &log2_a, unsigned &deg) {
        log2_a = constant_log2(f);
        deg = 0;
        for (unsigned i = 0; i < f.count; ++i) {
            const auto &x = f.factor[i];
            const long double term = x.b < 0 ? (long double)x.a - (long double)(uint64_t(-(x.b + 1)) + 1) / n
                                             : (long double)x.a;
            log2_a += x.power * log2l(term);
            deg += x.power;
        }
    };
    long double ap, aq, ar = 0;
    unsigned dp, dq, dr = 0;
    upper(def.P, ap, dp);
    lower(def.Q, aq, dq);
    if (common)
        upper(def.R, ar, dr);
    // rho_N bounds |R(j)/Q(j)| (or the BBP weight) for every j >= N.
    long double log2_rho = bbp ? -(long double)def.stride : ar - aq + (long double)(int(dr) - int(dq)) * log2l(n);
    const int d = int(dp) - int(dq);
    if (d > 0)
        log2_rho += (long double)d / (n * logl(2.0L)); // k^d <= N^d e^(d (k-N)/N)
    if (!(log2_rho < 0))
        return -1.0e30L; // no geometric tail from this N
    const long double rho = exp2l(log2_rho);
    // |sum_{k>=N}| <= A_P/A_Q N^d 2^-(att(N)) / (1 - rho); BBP adds the absolute weight of the first term.
    // Tail admission needs a bound on this entire prefix, not integer
    // increments. Keep attenuation_bits() for the conservative execution
    // policy, and use the tighter cumulative bound for the stopping index.
    long double log2_tail = ap - aq + (long double)d * log2l(n) - log2l(1 - rho) - attenuation_total(N);
    if (bbp)
        log2_tail += (long double)def.shift - (long double)def.stride * (long double)def.begin;
    // Rounding allowance: a few bits over the magnitude of the estimate.
    log2_tail += allowance(log2_tail);
    return -log2_tail;
}
uint64_t DataFormula::infinite_tail_terms(uint64_t bits_needed) const noexcept {
    if (tail_rejection || first_positive >= index_end)
        return 0;
    const long double need = (long double)bits_needed;
    if (infinite_tail_bits(index_end) < need)
        return 0;
    uint64_t lo = std::max<uint64_t>(first_positive, 1), hi = index_end;
    // The domain is usually a huge probe while the answer is small: gallop to
    // bracket it before bisecting (same answer for a monotone certificate, a
    // third of the evaluations).
    for (uint64_t step = 64; step < hi - lo; step *= 4) {
        const uint64_t m = lo + step;
        if (infinite_tail_bits(m) >= need) {
            hi = m;
            break;
        }
        lo = m + 1;
    }
    while (lo < hi) {
        const uint64_t m = lo + (hi - lo) / 2;
        // Not necessarily monotone near the start (rho may exceed 1 for small
        // N); treat a failing certificate as "need more terms".
        if (infinite_tail_bits(m) >= need)
            hi = m;
        else
            lo = m + 1;
    }
    return lo;
}
uint64_t DataFormula::tail_terms(uint64_t bits_needed) const noexcept {
    const long double need = (long double)bits_needed;
    if (attenuation_total(index_end) < need)
        return 0;
    uint64_t lo = def.begin, hi = index_end;
    while (lo < hi) {
        const uint64_t m = lo + (hi - lo) / 2;
        if (attenuation_total(m) >= need)
            hi = m;
        else
            lo = m + 1;
    }
    return lo;
}
void DataFormula::leaf(uint64_t k, unsigned need, sbn3_series_values &out) const noexcept {
    require(!rejection && k >= def.begin && k < index_end, SBN3_FATAL_ARGUMENT, "formula leaf index");
    const bool common = def.recipe == SBN3_SERIES_COMMON_P2B3, bbp = def.recipe == SBN3_SERIES_BINARY_BBP;
    const int64_t exponent = bbp ? def.shift - int64_t(uint64_t(def.stride) * k) : 0;
    if (def.explicit_first && k == def.begin) {
        if (need & 1)
            set(out.value[0], &def.first.t, 1, def.first.t_negative, exponent);
        if (need & 2)
            set(out.value[1], &def.first.d, 1, false, 0);
        if (need & 4)
            set(out.value[2], &def.first.u, 1, false, 0);
        return;
    }
    // Q's factors are positive on the domain; only its constant/(-1)^k can be
    // negative. Normalize to a positive denominator by negating every lane.
    const bool flip = def.Q.negative != (def.Q.alternating && (k & 1));
    uint64_t w[max_leaf_words];
    bool negative = false;
    if (need & 2) {
        size_t n = evaluate(def.Q, k, w, negative);
        require(n && negative == flip, SBN3_FATAL_MATH, "formula denominator sign");
        if (twos_shift) {
            // Exact: 2^twos_shift divides every factored Q(k).
            const size_t drop = twos_shift / 64;
            const unsigned bits_shift = twos_shift % 64;
            require(drop < n, SBN3_FATAL_MATH, "formula denominator valuation");
            for (size_t x = 0; x < drop; ++x)
                require(!w[x], SBN3_FATAL_MATH, "formula denominator valuation");
            require(!bits_shift || !(w[drop] & ((uint64_t(1) << bits_shift) - 1)), SBN3_FATAL_MATH,
                    "formula denominator valuation");
            for (size_t x = 0; x + drop < n; ++x)
                w[x] = bits_shift ? (w[x + drop] >> bits_shift) | (x + drop + 1 < n ? w[x + drop + 1] << (64 - bits_shift) : 0)
                                  : w[x + drop];
            n -= drop;
        }
        set(out.value[1], w, n, false, 0);
    }
    if (need & 1) {
        const size_t n = evaluate(def.P, k, w, negative);
        set(out.value[0], w, n, negative != flip, exponent - int64_t(twos_shift));
    }
    if ((need & 4) && common) {
        const size_t n = evaluate(def.R, k, w, negative);
        set(out.value[2], w, n, negative != flip, -int64_t(twos_shift));
    }
}
namespace {
constexpr size_t wide_batch_limbs = 64 * 3 + 8;
// v <- v * x for a 128-bit x, in place; scratch holds a copy for the two-limb case.
void times_wide(sbn3_int &v, u128 x, uint64_t *scratch) {
    if (!x || !v.size) {
        v.size = 0;
        v.negative = 0;
        return;
    }
    const uint64_t lo = uint64_t(x), hi = uint64_t(x >> 64);
    if (!hi) {
        const uint64_t carry = sbn3i_mul_1(v.data, v.data, long(v.size), lo);
        if (carry) {
            require(v.size < v.capacity, SBN3_FATAL_MATH, "wide batch capacity");
            v.data[v.size++] = carry;
        }
        return;
    }
    const size_t n = v.size;
    require(n <= wide_batch_limbs && n + 2 <= v.capacity, SBN3_FATAL_MATH, "wide batch capacity");
    std::memcpy(scratch, v.data, n * 8);
    v.data[n] = sbn3i_mul_1(v.data, scratch, long(n), lo);
    v.data[n + 1] = sbn3i_addmul_1(v.data + 1, scratch, long(n), hi);
    v.size = n + 2;
    while (v.size && !v.data[v.size - 1])
        --v.size;
}
} // namespace
void DataFormula::batch(sbn3_series_range r, unsigned need, sbn3_series_values &out) const noexcept {
    if (leaf_kind == LeafKind::CommonWideBatch) {
        require(r.begin >= def.begin && r.begin < r.end && r.end <= index_end && r.end - r.begin <= 64,
                SBN3_FATAL_ARGUMENT, "common wide batch interval");
        uint64_t local_u[wide_batch_limbs], product_words[wide_batch_limbs], scratch[wide_batch_limbs];
        sbn3_int local{local_u, wide_batch_limbs, 1, 0};
        local_u[0] = 1;
        auto &U = need & 4 ? out.value[2].mantissa : local;
        auto &T = out.value[0].mantissa;
        auto &D = out.value[1].mantissa;
        const bool want_u = need & 5;
        if (need & 1) {
            T.size = 0;
            T.negative = 0;
            out.value[0].exponent2 = 0;
        }
        if (need & 2) {
            D.size = 1;
            D.negative = 0;
            D.data[0] = 1;
            out.value[1].exponent2 = 0;
        }
        if (want_u) {
            U.size = 1;
            U.negative = 0;
            U.data[0] = 1;
            out.value[2].exponent2 = 0;
        }
        // value = constant * index part; both below 2^128 (admission).
        struct Wide {
            u128 constant, index;
            bool negative;
        };
        auto wide = [](const FactorProduct &f, uint64_t k) {
            Wide w{constant(f), 1, f.negative != (f.alternating && (k & 1))};
            for (unsigned j = 0; j < f.count; ++j) {
                const auto y = factor_value(f.factor[j], k);
                const auto magnitude = u128(y < 0 ? -y : y);
                if (y < 0 && (f.factor[j].power & 1))
                    w.negative = !w.negative;
                for (unsigned p = 0; p < f.factor[j].power; ++p)
                    w.index *= magnitude;
            }
            if (f.degree) {
                i128 y = f.coefficient[f.degree];
                for (unsigned j = f.degree; j-- > 0;)
                    y = y * i128(k) + f.coefficient[j];
                if (y < 0) {
                    w.negative = !w.negative;
                    y = -y;
                }
                w.index *= u128(y);
            }
            return w;
        };
        auto apply = [&](sbn3_int &v, const Wide &w) {
            if (bits(w.constant) + bits(w.index) <= 128)
                times_wide(v, w.constant * w.index, scratch);
            else {
                times_wide(v, w.index, scratch);
                times_wide(v, w.constant, scratch);
            }
        };
        for (uint64_t k = r.begin; k < r.end; ++k) {
            Wide p{0, 1, false}, q{1, 1, false}, u{1, 1, false};
            if (def.explicit_first && k == def.begin) {
                p = {def.first.t, 1, def.first.t_negative};
                q = {def.first.d, 1, false};
                u = {def.first.u, 1, false};
            } else {
                q = wide(def.Q, k);
                if (need & 1)
                    p = wide(def.P, k);
                if (want_u)
                    u = wide(def.R, k);
                p.negative ^= q.negative;
                u.negative ^= q.negative;
            }
            require(q.constant && q.index, SBN3_FATAL_MATH, "common wide batch nonzero denominator");
            if (need & 1) {
                apply(T, q);
                sbn3_int product{product_words, wide_batch_limbs, 0, 0};
                if (p.constant && p.index && U.size) {
                    require(U.size <= wide_batch_limbs - 4, SBN3_FATAL_MATH, "wide batch product capacity");
                    std::memcpy(product.data, U.data, U.size * 8);
                    product.size = U.size;
                    product.negative = U.negative ^ unsigned(p.negative);
                    apply(product, p);
                }
                sbn3_int_add(&T, {T.data, T.size, T.negative}, {product.data, product.size, product.negative});
            }
            if (need & 2)
                apply(D, q);
            if (want_u) {
                apply(U, u);
                if (U.size)
                    U.negative ^= unsigned(u.negative);
            }
        }
        return;
    }
    if (leaf_kind == LeafKind::CommonWordBatch) {
        require(r.begin >= def.begin && r.begin < r.end && r.end <= index_end && r.end - r.begin <= 64,
                SBN3_FATAL_ARGUMENT, "common word batch interval");
        uint64_t local_u[66], product_words[66];
        sbn3_int local{local_u, 66, 1, 0};
        local_u[0] = 1;
        auto &U = need & 4 ? out.value[2].mantissa : local;
        auto &T = out.value[0].mantissa;
        auto &D = out.value[1].mantissa;
        const bool want_u = need & 5;
        if (need & 1) {
            T.size = 0;
            T.negative = 0;
            out.value[0].exponent2 = 0;
        }
        if (need & 2) {
            D.size = 1;
            D.negative = 0;
            D.data[0] = 1;
            out.value[1].exponent2 = 0;
        }
        if (want_u) {
            U.size = 1;
            U.negative = 0;
            U.data[0] = 1;
            out.value[2].exponent2 = 0;
        }
        auto append = [](sbn3_int &v, uint64_t carry) {
            if (carry) {
                require(v.size < v.capacity, SBN3_FATAL_MATH, "common batch capacity");
                v.data[v.size++] = carry;
            }
        };
        auto multiply = [&](sbn3_int &v, uint64_t x, bool negative) {
            if (!x || !v.size) {
                v.size = 0;
                v.negative = 0;
                return;
            }
            append(v, sbn3i_mul_1(v.data, v.data, long(v.size), x));
            v.negative ^= unsigned(negative);
        };
        auto word = [](const FactorProduct &f, uint64_t k, bool &negative) {
            u128 x = constant(f);
            negative = f.negative != (f.alternating && (k & 1));
            for (unsigned j = 0; j < f.count; ++j) {
                const auto y = factor_value(f.factor[j], k);
                const auto magnitude = u128(y < 0 ? -y : y);
                if (y < 0 && (f.factor[j].power & 1))
                    negative = !negative;
                for (unsigned p = 0; p < f.factor[j].power; ++p)
                    x *= magnitude;
            }
            if (f.degree) {
                i128 y = f.coefficient[f.degree];
                for (unsigned j = f.degree; j-- > 0;)
                    y = y * i128(k) + f.coefficient[j];
                if (y < 0) {
                    negative = !negative;
                    y = -y;
                }
                x *= u128(y);
            }
            require(!(x >> 64), SBN3_FATAL_MATH, "common batch word bound");
            return uint64_t(x);
        };
        for (uint64_t k = r.begin; k < r.end; ++k) {
            bool pn = false, qn = false, un = false;
            uint64_t p = 0, q = 0, u = 1;
            if (def.explicit_first && k == def.begin) {
                p = def.first.t;
                q = def.first.d;
                u = def.first.u;
                pn = def.first.t_negative;
            } else {
                q = word(def.Q, k, qn);
                if (need & 1)
                    p = word(def.P, k, pn);
                if (want_u)
                    u = word(def.R, k, un);
                pn ^= qn;
                un ^= qn;
            }
            require(q, SBN3_FATAL_MATH, "common batch nonzero denominator");
            if (need & 1) {
                multiply(T, q, false);
                sbn3_int product{product_words, 66, 0, 0};
                if (p && U.size) {
                    product.size = U.size;
                    product.negative = U.negative ^ unsigned(pn);
                    append(product, sbn3i_mul_1(product.data, U.data, long(U.size), p));
                }
                sbn3_int_add(&T, {T.data, T.size, T.negative},
                             {product.data, product.size, product.negative});
            }
            if (need & 2)
                multiply(D, q, false);
            if (want_u)
                multiply(U, u, un);
        }
        return;
    }
    require(leaf_kind == LeafKind::HyperWordBatch && r.begin >= def.begin && r.begin < r.end && r.end <= index_end,
            SBN3_FATAL_ARGUMENT, "formula batch interval");
    auto &T = out.value[0].mantissa;
    auto &D = out.value[1].mantissa;
    bool t_negative = false;
    uint64_t k = r.begin;
    if (need & 1) {
        require(T.capacity, SBN3_FATAL_WORKSPACE, "formula batch output");
        T.size = 0;
        out.value[0].exponent2 = 0;
    }
    if (need & 2) {
        require(D.capacity, SBN3_FATAL_WORKSPACE, "formula batch output");
        D.size = 1;
        D.negative = 0;
        D.data[0] = 1;
        out.value[1].exponent2 = 0;
    }
    if (def.explicit_first && k == def.begin) {
        if (need & 1) {
            T.data[0] = def.first.t;
            T.size = def.first.t ? 1 : 0;
            t_negative = def.first.t_negative && def.first.t;
        }
        if (need & 2)
            D.data[0] = def.first.d;
        ++k;
    }
    auto append = [](sbn3_int &v, uint64_t carry) {
        if (carry) {
            require(v.size < v.capacity, SBN3_FATAL_MATH, "formula batch bound");
            v.data[v.size++] = carry;
        }
    };
    for (; k < r.end; ++k) {
        u128 qv = constant(def.Q);
        for (unsigned i = 0; i < def.Q.count; ++i) {
            const u128 f = u128(factor_value(def.Q.factor[i], k));
            for (unsigned e = 0; e < def.Q.factor[i].power; ++e)
                qv *= f;
        }
        require(qv && !(qv >> 64), SBN3_FATAL_MATH, "formula batch denominator word");
        const uint64_t q = uint64_t(qv);
        if (need & 2)
            append(D, sbn3i_mul_1(D.data, D.data, long(D.size), q));
        if (!(need & 1))
            continue;
        bool s = def.P.negative != (def.P.alternating && (k & 1));
        uint64_t p = def.P.constant_low;
        if (def.P.degree) {
            // Admission proved |constant * polynomial(k)| < 2^63 on the domain.
            i128 value = def.P.coefficient[def.P.degree];
            for (unsigned i = def.P.degree; i-- > 0;)
                value = value * i128(k) + def.P.coefficient[i];
            if (value < 0) {
                s = !s;
                value = -value;
            }
            const u128 scaled = u128(value) * p;
            require(!(scaled >> 63), SBN3_FATAL_MATH, "formula batch numerator word");
            p = uint64_t(scaled);
        }
        if (!T.size) {
            T.data[0] = p;
            T.size = p ? 1 : 0;
            t_negative = s && p;
            continue;
        }
        if (t_negative == s || !p) {
            append(T, sbn3i_mul_1c(T.data, T.data, long(T.size), q, p));
            continue;
        }
        append(T, sbn3i_mul_1(T.data, T.data, long(T.size), q));
        if (T.size == 1 && T.data[0] < p) {
            T.data[0] = p - T.data[0];
            t_negative = s;
            continue;
        }
        uint64_t borrow = T.data[0] < p;
        T.data[0] -= p;
        for (size_t j = 1; borrow && j < T.size; ++j) {
            borrow = T.data[j] == 0;
            --T.data[j];
        }
        require(!borrow, SBN3_FATAL_MATH, "formula batch borrow");
        while (T.size && !T.data[T.size - 1])
            --T.size;
        if (!T.size)
            t_negative = false;
    }
    if (need & 1)
        T.negative = t_negative && T.size;
}
FiniteFormula data_finite_formula(const DataFormula &d) noexcept {
    require(!d.rejection && d.index_end && !d.probe_only, SBN3_FATAL_ARGUMENT, "data formula not prepared");
    FiniteFormula f{&d,
                    d.def.recipe,
                    d.formula_id,
                    d.parameter_id,
                    d.leaf_words,
                    [](const void *p, sbn3_series_range r, uint64_t n, unsigned need, sbn3_series_shape *s) {
                        return static_cast<const DataFormula *>(p)->bounds(r, n, need, *s);
                    },
                    [](const void *p, sbn3_series_range r) { return static_cast<const DataFormula *>(p)->work(r); },
                    [](const void *p, uint64_t k, unsigned need, sbn3_series_values *v) {
                        static_cast<const DataFormula *>(p)->leaf(k, need, *v);
                    },
                    d.leaf_kind != LeafKind::Factored
                        ? +[](const void *p, sbn3_series_range r, unsigned need, sbn3_series_values *v) {
                              static_cast<const DataFormula *>(p)->batch(r, need, *v);
                          }
                        : nullptr};
    if(f.batch){f.batch_words_per_term=d.batch_words;f.batch_max_terms=64;}
    f.exact_span = [](const void *p, sbn3_series_range r, unsigned need, sbn3_series_shape *s) {
        const auto &d = *static_cast<const DataFormula *>(p);
        return d.bounds(r, r.end - r.begin, need, *s, d.Q.twos && !d.twos_shift, true);
    };
    if (d.smooth_split) {
        f.split_policy_id = 0x444154414d415331ULL; // DATAMAS1: equal integer-measure serial cuts
        f.split_point = [](const void *p, sbn3_series_range r, double fraction) {
            return static_cast<const DataFormula *>(p)->split_point(r, fraction);
        };
        f.serial_envelope = [](const void *p, sbn3_series_range r, unsigned depth, unsigned need,
                               uint64_t *max_terms, sbn3_series_shape *s) {
            return static_cast<const DataFormula *>(p)->serial_envelope(r, depth, need, *max_terms, *s);
        };
    }
    if (d.Q.twos && !d.twos_shift) {
        f.normalized_bounds = [](const void *p, sbn3_series_range r, uint64_t n, unsigned need,
                                 sbn3_series_shape *s) {
            return static_cast<const DataFormula *>(p)->bounds(r, n, need, *s, true);
        };
        if (d.smooth_split)
            f.normalized_serial_envelope = [](const void *p, sbn3_series_range r, unsigned depth, unsigned need,
                                              uint64_t *max_terms, sbn3_series_shape *s) {
                return static_cast<const DataFormula *>(p)->serial_envelope(r, depth, need, *max_terms, *s, true);
            };
    }
    return f;
}
FormulaDef euler_definition() noexcept {
    FormulaDef d{};
    d.recipe = SBN3_SERIES_HYPERDESCENT;
    d.begin = 1;
    d.Q.count = 1;
    d.Q.factor[0] = {1, 0, 1};
    return d;
}
FormulaDef chudnovsky_definition() noexcept {
    constexpr uint64_t A = 13591409, B = 545140134, K = 10939058860032000ULL;
    FormulaDef d{};
    d.recipe = SBN3_SERIES_COMMON_P2B3;
    d.begin = 0;
    d.explicit_first = true;
    d.first = {A, 1, 1, false};
    d.P.alternating = true;
    d.P.count = 4;
    d.P.factor[0] = {6, -5, 1};
    d.P.factor[1] = {2, -1, 1};
    d.P.factor[2] = {6, -1, 1};
    d.P.factor[3] = {B, int64_t(A), 1};
    d.Q.constant_low = K;
    d.Q.count = 1;
    d.Q.factor[0] = {1, 0, 3};
    d.R.count = 3;
    d.R.factor[0] = {6, -5, 1};
    d.R.factor[1] = {2, -1, 1};
    d.R.factor[2] = {6, -1, 1};
    return d;
}
FormulaDef binary_log_definition(unsigned radix_bits) noexcept {
    FormulaDef d{};
    d.recipe = SBN3_SERIES_BINARY_BBP;
    d.begin = 1;
    d.Q.count = 1;
    d.Q.factor[0] = {1, 0, 1};
    d.shift = 0;
    d.stride = radix_bits;
    return d;
}
FormulaDef exp_reciprocal_definition(uint64_t m) noexcept {
    FormulaDef d{};
    d.recipe = SBN3_SERIES_HYPERDESCENT;
    d.begin = 0;
    d.explicit_first = true;
    d.first = {1, 1, 1, false};
    d.Q.constant_low = m;
    d.Q.count = 1;
    d.Q.factor[0] = {1, 0, 1};
    return d;
}
FormulaDef exp_minus_one_definition() noexcept {
    FormulaDef d{};
    d.recipe = SBN3_SERIES_HYPERDESCENT;
    d.begin = 0;
    d.explicit_first = true;
    d.first = {1, 1, 1, false};
    d.P.alternating = true;
    d.Q.count = 1;
    d.Q.factor[0] = {1, 0, 1};
    return d;
}
} // namespace sbn::v3::series
