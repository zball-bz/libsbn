#include "common/checked.hpp"
#include "common/identity.hpp"
#include "sbn3/log.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace sbn::v3::series {
namespace {
// Relation components atanh(a/b) = (1/2) log((b+a)/(b-a)), b+a and b-a {2,3,5,7}-smooth.
// Integer arguments (a = 1): all 2<=m<=10000 with m-1 and m+1 smooth, reproduced independently by
// bench/probes/log_formula_search_2026-09-17/search.py. Rational arguments: every representable
// pair that enters a best relation of the 107 targets 2^i 3^j 5^k 7^l (i<=3, j,k,l<=2) in the
// exhaustive screening bench/probes/log_followup_2026-09-17/rational_catalog.py (all coprime smooth
// s>t up to 70000, same ranking objective). Their (b+a)/(b-a) are the small smooth commas
// 128/125, 3136/3125, 256/243, 16875/16807, 525/512, 15625/15552, 256/245, 250/243, 5120/5103,
// 3125/3087 and 25/21.
struct Argument {
    uint64_t numerator, denominator;
};
constexpr Argument arguments[]{{1, 8749}, {1, 4801},    {1, 449},    {1, 251},     {1, 244},   {1, 161},
                               {1, 127},  {1, 99},      {1, 97},     {1, 71},      {1, 55},    {1, 49},
                               {1, 41},   {1, 31},      {1, 29},     {1, 26},      {1, 19},    {1, 17},
                               {1, 15},   {1, 13},      {1, 11},     {1, 9},       {1, 8},     {1, 7},
                               {1, 6},    {1, 5},       {1, 4},      {1, 3},       {1, 2},     {3, 253},
                               {11, 6261}, {13, 499},   {34, 16841}, {13, 1037},   {73, 31177}, {11, 501},
                               {7, 493},  {17, 10223},  {19, 3106},  {2, 23}};
constexpr unsigned catalog_size = sizeof(arguments) / sizeof(*arguments);
struct Candidate {
    uint64_t numerator, argument;
    int64_t vector[4];
    double cost;
};
using u128 = __uint128_t;
u128 gcd128(u128 a, u128 b) {
    while (b) {
        const u128 t = a % b;
        a = b;
        b = t;
    }
    return a;
}
// Exact integer data of the accelerated series of atanh(a/b) (see sbn3/log.h).
// Returns false when the data leaves the schema (polynomial coefficients of P
// within int64, constants within 128 bits).
struct AcceleratedData {
    u128 q = 0, r = 0, linear = 0, constant = 0, divisor = 0;
};
bool accelerated_data(uint64_t a, uint64_t b, AcceleratedData &d) {
    if (!a || b > 65536 || u128(a) * 2 > b)
        return false; // keeps 72 B^2 and 3 B (B-A)^2 inside 128 bits
    const u128 A = u128(a) * a, B = u128(b) * b;
    d.q = 3 * B * (B - A) * (B - A);
    d.r = 8 * A * A * A;
    d.linear = 8 * (9 * B * B - 15 * A * B + 4 * A * A);
    d.constant = 4 * B * (3 * B - 5 * A);
    // For a = 1 this is gcd(8, 3x(x-1)^2): 4 or 8.
    d.divisor = gcd128(gcd128(d.q, d.r), gcd128(d.linear, d.constant));
    d.q /= d.divisor;
    d.r /= d.divisor;
    d.linear /= d.divisor;
    d.constant /= d.divisor;
    return d.linear <= u128(INT64_MAX) && d.constant <= u128(INT64_MAX);
}
bool accelerated(uint64_t a, uint64_t b, sbn3_formula_def *out) {
    AcceleratedData d;
    if (!accelerated_data(a, b, d))
        return false;
    if (!out)
        return true;
    sbn3_formula_def f{};
    f.recipe = SBN3_SERIES_COMMON_P2B3;
    f.begin = 1;
    const u128 ab = u128(a) * b;
    f.P.constant_low = uint64_t(ab);
    f.P.constant_high = uint64_t(ab >> 64);
    f.P.degree = 1;
    f.P.coefficient[0] = -int64_t(d.constant);
    f.P.coefficient[1] = int64_t(d.linear);
    f.Q.constant_low = uint64_t(d.q);
    f.Q.constant_high = uint64_t(d.q >> 64);
    f.Q.factor_count = 2;
    f.Q.factor[0] = {6, -1, 1};
    f.Q.factor[1] = {6, -5, 1};
    f.R.constant_low = uint64_t(d.r);
    f.R.constant_high = uint64_t(d.r >> 64);
    f.R.factor_count = 2;
    f.R.factor[0] = {1, 0, 1};
    f.R.factor[1] = {2, -1, 1};
    f.denominator_exponent = 2; // outer factor 1/4
    *out = f;
    return true;
}
void taylor(uint64_t a, uint64_t b, sbn3_formula_def *out) {
    sbn3_formula_def f{};
    f.P.constant_low = f.R.constant_low = 1;
    f.Q.factor_count = 1;
    f.Q.factor[0] = {2, 1, 1};
    if (a == 1 && !(b & (b - 1))) {
        const unsigned s = __builtin_ctzll(b);
        f.recipe = SBN3_SERIES_BINARY_BBP;
        f.Q.constant_low = 1;
        f.shift = -int64_t(s);
        f.stride = 2 * s;
    } else {
        // sum_{k>=0} (a/b)^(2k+1)/(2k+1): P = a b, Q = b^2 (2k+1), R = a^2 (2k+1).
        const u128 square = u128(b) * b, top = u128(a) * b, low = u128(a) * a;
        f.recipe = SBN3_SERIES_COMMON_P2B3;
        f.P.constant_low = uint64_t(top);
        f.P.constant_high = uint64_t(top >> 64);
        f.Q.constant_low = uint64_t(square);
        f.Q.constant_high = uint64_t(square >> 64);
        f.R.constant_low = uint64_t(low);
        f.R.constant_high = uint64_t(low >> 64);
        f.R.factor_count = 1;
        f.R.factor[0] = {2, 1, 1};
    }
    *out = f;
}
// Ranking objective of the relation search: modelled binary-splitting work of
// ArcCoth(m) with its AUTO series, per bit of requested precision, at one
// fixed reference precision (2^26 bits) so the chosen formula does not depend
// on the request. Per term a Common merge level writes about 2.5 gQ + 1.5 gR
// bits (T D, D D, U T, U U) and a BinaryBBP level 3 gQ, where gQ and gR are the
// mean bit growth of Q and R; the term count is precision / convergence bits.
// `fixed` charges one terminal division and binding per component. Measured
// on Zen5 at 2^20 limbs the model tracks series+merge time within about 10%
// across m = 2..8749; it only ranks candidates and never affects correctness.
// A rational argument a/b converges like the integer b/a but carries the larger
// constants 3B(B-A)^2 and 8A^3 (A = a^2, B = b^2) in Q and R: about 20% more
// work per bit at b/a = 84, which a relation with fewer components can repay.
double series_work(uint64_t a, uint64_t m) {
    constexpr double reference_bits = 67108864.0, fixed = 1.7, mean = 1.4426950408889634; // 1/ln 2
    if (a != 1) {
        AcceleratedData d;
        require(accelerated_data(a, m, d), SBN3_FATAL_MATH, "log catalog rational argument");
        const auto lg = [](u128 x) { return std::log2(double(uint64_t(x >> 64)) * 18446744073709551616.0 + double(uint64_t(x))); };
        const double A = double(a) * double(a), B = double(m) * double(m);
        const double bits = std::log2(6.75) + std::log2(B) + 2 * std::log2(B - A) - 3 * std::log2(A);
        const double terms = reference_bits / bits;
        const double q = lg(d.q) + 2 * (std::log2(6 * terms) - mean);
        const double r = lg(d.r) + std::log2(terms) + std::log2(2 * terms) - 2 * mean;
        return (2.5 * q + 1.5 * r) / bits + fixed;
    }
    const double lm = std::log2(double(m));
    if (accelerated(1, m, nullptr)) {
        const double lx = 2 * lm, lx1 = std::log2(double(m - 1)) + std::log2(double(m + 1));
        const double lg = (m & 1) || !(m & 3) ? 3 : 2;
        const double bits = std::log2(6.75) + lx + 2 * lx1, terms = reference_bits / bits;
        const double q = std::log2(3.0) + lx + 2 * lx1 - lg + 2 * (std::log2(6 * terms) - mean);
        const double r = 3 - lg + std::log2(terms) + std::log2(2 * terms) - 2 * mean;
        return (2.5 * q + 1.5 * r) / bits + fixed;
    }
    const double bits = 2 * lm, odd = std::log2(2 * reference_bits / bits) - mean;
    if (!(m & (m - 1)))
        return 3 * odd / bits + fixed;
    return (2.5 * (bits + odd) + 1.5 * odd) / bits + fixed;
}
bool factors(uint64_t n, int64_t *out) {
    constexpr unsigned primes[]{2, 3, 5, 7};
    for (unsigned j = 0; j < 4; ++j) {
        out[j] = 0;
        while (n % primes[j] == 0) {
            n /= primes[j];
            ++out[j];
        }
    }
    return n == 1;
}
void catalog(Candidate *out) {
    for (unsigned j = 0; j < catalog_size; ++j) {
        auto &c = out[j];
        c.numerator = arguments[j].numerator;
        c.argument = arguments[j].denominator;
        c.cost = series_work(c.numerator, c.argument);
        int64_t hi[4], lo[4];
        require(std::gcd(c.numerator, c.argument) == 1 && factors(c.argument + c.numerator, hi) &&
                    factors(c.argument - c.numerator, lo),
                SBN3_FATAL_MATH, "log relation catalog");
        for (unsigned k = 0; k < 4; ++k)
            c.vector[k] = hi[k] - lo[k];
    }
    // The search prunes on ascending cost.
    std::sort(out, out + catalog_size, [](const Candidate &a, const Candidate &b) {
        return a.cost < b.cost || (a.cost == b.cost && a.argument > b.argument);
    });
}
uint64_t magnitude(int64_t x) { return x < 0 ? uint64_t(-(x + 1)) + 1 : uint64_t(x); }
int64_t determinant(const int64_t a[3][3], unsigned n) {
    if (n == 1)
        return a[0][0];
    if (n == 2)
        return a[0][0] * a[1][1] - a[0][1] * a[1][0];
    return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
           a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
           a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}
bool relation(const Candidate *const *columns, unsigned width, const int64_t *target, int64_t *coefficients,
              uint64_t &divisor) {
    unsigned rows[3]{};
    int64_t matrix[3][3]{}, denominator = 0;
    for (unsigned mask = 1; mask < 16 && !denominator; ++mask)
        if (unsigned(__builtin_popcount(mask)) == width) {
            unsigned at = 0;
            for (unsigned r = 0; r < 4; ++r)
                if (mask & (1u << r))
                    rows[at++] = r;
            for (unsigned i = 0; i < width; ++i)
                for (unsigned j = 0; j < width; ++j)
                    matrix[i][j] = columns[j]->vector[rows[i]];
            denominator = determinant(matrix, width);
        }
    if (!denominator)
        return false;
    int64_t numerators[3]{};
    for (unsigned j = 0; j < width; ++j) {
        int64_t copy[3][3];
        std::memcpy(copy, matrix, sizeof copy);
        for (unsigned i = 0; i < width; ++i)
            copy[i][j] = target[rows[i]];
        numerators[j] = determinant(copy, width);
        if (!numerators[j])
            return false;
    }
    for (unsigned r = 0; r < 4; ++r) {
        int64_t sum = 0;
        for (unsigned j = 0; j < width; ++j)
            sum += columns[j]->vector[r] * numerators[j];
        if (sum != target[r] * denominator)
            return false;
    }
    uint64_t g = magnitude(denominator);
    for (unsigned j = 0; j < width; ++j)
        g = std::gcd(g, magnitude(numerators[j]));
    divisor = magnitude(denominator) / g;
    if (divisor > 8)
        return false;
    for (unsigned j = 0; j < width; ++j) {
        coefficients[j] = numerators[j] / int64_t(g) * (denominator < 0 ? -1 : 1);
        if (magnitude(coefficients[j]) > 128)
            return false;
    }
    return true;
}
void normalize(sbn3_log_formula &f, uint32_t n) {
    unsigned count = 0;
    uint64_t g = f.divisor;
    for (unsigned j = 0; j < f.count; ++j)
        if (f.term[j].coefficient) {
            f.term[count++] = f.term[j];
            g = std::gcd(g, magnitude(f.term[j].coefficient));
        }
    f.count = count;
    for (unsigned j = 0; j < count; ++j)
        f.term[j].coefficient /= int64_t(g);
    f.divisor /= g;
    std::sort(f.term, f.term + count, [](const auto &a, const auto &b) {
        return a.argument < b.argument || (a.argument == b.argument && a.numerator < b.numerator);
    });
    f.lehmer_measure = f.work_estimate = 0;
    uint64_t id = identity::word(identity::word(identity::fnv_seed, n), f.divisor);
    for (unsigned j = 0; j < count; ++j) {
        f.lehmer_measure += 1 / (2 * std::log10(double(f.term[j].argument) / double(f.term[j].numerator)));
        f.work_estimate += series_work(f.term[j].numerator, f.term[j].argument);
        id = identity::word(identity::word(id, f.term[j].argument), uint64_t(f.term[j].coefficient));
        if (f.term[j].numerator != 1) // integer-argument identities are unchanged
            id = identity::word(id, f.term[j].numerator);
    }
    f.identity = id;
}
void add(sbn3_log_formula &f, uint64_t argument, int64_t coefficient, uint64_t numerator = 1) {
    for (unsigned j = 0; j < f.count; ++j)
        if (f.term[j].argument == argument && f.term[j].numerator == numerator) {
            f.term[j].coefficient += coefficient;
            return;
        }
    require(f.count < SBN3_LOG_MAX_TERMS, SBN3_FATAL_MATH, "log argument count");
    f.term[f.count++] = {argument, coefficient, numerator};
}
void search(const Candidate *c, const int64_t *target, sbn3_log_formula &best) {
    // Small fixed search domain. Positive support costs give a lower bound;
    // coefficients of every accepted support are nonzero, so no later
    // cancellation can invalidate this pruning within a support.
    for (unsigned width = 1; width <= 3; ++width) {
        for (unsigned a = 0; a < catalog_size; ++a) {
            if (c[a].cost >= best.work_estimate)
                break;
            const unsigned b_start = width > 1 ? a + 1 : catalog_size;
            for (unsigned b = b_start; b <= catalog_size; ++b) {
                if (width > 1 && b == catalog_size)
                    break;
                const double ab = c[a].cost + (width > 1 ? c[b].cost : 0);
                if (ab >= best.work_estimate)
                    break;
                const unsigned d_start = width > 2 ? b + 1 : catalog_size;
                for (unsigned d = d_start; d <= catalog_size; ++d) {
                    if (width > 2 && d == catalog_size)
                        break;
                    const double cost = ab + (width > 2 ? c[d].cost : 0);
                    if (cost >= best.work_estimate)
                        break;
                    const Candidate *columns[]{&c[a], width > 1 ? &c[b] : nullptr,
                                               width > 2 ? &c[d] : nullptr};
                    int64_t coefficients[3]{};
                    uint64_t divisor = 0;
                    if (relation(columns, width, target, coefficients, divisor)) {
                        best = {};
                        best.divisor = divisor;
                        best.count = width;
                        best.work_estimate = cost;
                        for (unsigned j = 0; j < width; ++j)
                            best.term[j] = {columns[j]->argument, coefficients[j], columns[j]->numerator};
                    }
                    if (width < 3)
                        break;
                }
                if (width < 2)
                    break;
            }
        }
    }
}
} // namespace
} // namespace sbn::v3::series
using namespace sbn::v3;
using namespace sbn::v3::series;
extern "C" sbn3_query_result sbn3_atanh_series_definition(uint64_t numerator, uint64_t denominator,
                                                          sbn3_arccoth_series series, sbn3_formula_def *out) {
    require(out, SBN3_FATAL_ARGUMENT, "atanh definition output");
    if (!numerator || denominator / 2 < numerator || std::gcd(numerator, denominator) != 1 ||
        series < SBN3_ARCCOTH_AUTO || series > SBN3_ARCCOTH_ACCELERATED)
        return SBN3_UNSUPPORTED;
    if (series != SBN3_ARCCOTH_TAYLOR && accelerated(numerator, denominator, out))
        return SBN3_SUPPORTED;
    if (series == SBN3_ARCCOTH_ACCELERATED)
        return SBN3_UNSUPPORTED;
    taylor(numerator, denominator, out);
    return SBN3_SUPPORTED;
}
extern "C" sbn3_query_result sbn3_arccoth_series_definition(uint64_t m, sbn3_arccoth_series series,
                                                            sbn3_formula_def *out) {
    return sbn3_atanh_series_definition(1, m, series, out);
}
extern "C" sbn3_query_result sbn3_arccoth_definition(uint64_t m, sbn3_formula_def *out) {
    return sbn3_arccoth_series_definition(m, SBN3_ARCCOTH_AUTO, out);
}
extern "C" sbn3_query_result sbn3_arccoth_query(uint64_t m, size_t n, const sbn3_formula_options *options,
                                                void *object, sbn3_formula_plan *out,
                                                sbn3_formula_info *info) {
    require(out && info, SBN3_FATAL_ARGUMENT, "arccoth query output");
    sbn3_formula_def f{};
    if (sbn3_arccoth_definition(m, &f) != SBN3_SUPPORTED) {
        *info = {};
        info->rejection = "real positive ArcCoth requires an integer >=2";
        return SBN3_UNSUPPORTED;
    }
    return sbn3_formula_query(&f, n, options, object, out, info);
}
extern "C" sbn3_query_result sbn3_log_formula_query(uint32_t n, sbn3_log_formula *out) {
    require(out, SBN3_FATAL_ARGUMENT, "log formula output");
    if (!n)
        return SBN3_UNSUPPORTED;
    sbn3_log_formula result{};
    result.divisor = 1;
    if (n == 1) {
        normalize(result, n);
        *out = result;
        return SBN3_SUPPORTED;
    }
    Candidate candidates[catalog_size];
    catalog(candidates);
    sbn3_log_formula base{};
    base.divisor = 1;
    base.count = 1;
    base.term[0] = {3, 1, 1};
    base.work_estimate = series_work(1, 3);
    const int64_t two[]{1, 0, 0, 0};
    search(candidates, two, base);
    result.divisor = base.divisor;
    unsigned powers = 0;
    for (uint32_t k = n; k > 1; k >>= 1) {
        if (k & 1)
            add(result, 2 * uint64_t(k) - 1, int64_t(result.divisor));
        ++powers;
    }
    for (unsigned j = 0; j < base.count; ++j)
        add(result, base.term[j].argument, base.term[j].coefficient * powers);
    normalize(result, n);
    int64_t target[4]{};
    if (factors(n, target))
        search(candidates, target, result);
    normalize(result, n);
    *out = result;
    return SBN3_SUPPORTED;
}
extern "C" size_t sbn3_log_object_bytes(uint32_t n) {
    sbn3_log_formula f{};
    return sbn3_log_formula_query(n, &f) == SBN3_SUPPORTED ? sbn3_formula_sum_object_bytes(f.count) : 0;
}
extern "C" sbn3_query_result sbn3_log_query(uint32_t n, size_t limbs, const sbn3_formula_options *options,
                                            void *object, size_t capacity, sbn3_formula_sum_plan *plan,
                                            sbn3_formula_sum_info *info) {
    require(plan && info, SBN3_FATAL_ARGUMENT, "log query output");
    sbn3_log_formula f{};
    if (sbn3_log_formula_query(n, &f) != SBN3_SUPPORTED) {
        *info = {};
        info->rejection = "integer Log requires n>=1";
        return SBN3_UNSUPPORTED;
    }
    sbn3_formula_sum_term terms[SBN3_LOG_MAX_TERMS]{};
    for (unsigned j = 0; j < f.count; ++j) {
        require(sbn3_atanh_series_definition(f.term[j].numerator, f.term[j].argument, SBN3_ARCCOTH_AUTO,
                                             &terms[j].formula) == SBN3_SUPPORTED,
                SBN3_FATAL_MATH, "log atanh argument");
        terms[j].coefficient = 2 * f.term[j].coefficient;
    }
    return sbn3_formula_sum_query(terms, f.count, f.divisor, limbs, options, object, capacity, plan, info);
}
