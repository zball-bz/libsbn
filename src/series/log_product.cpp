#include "series/log_product.hpp"
#include "common/checked.hpp"
#include <cfloat>
#include <cmath>
namespace sbn::v3::series {
namespace {
using u128 = __uint128_t;
constexpr double ln2 = 0.693147180559945309417232121458176568;
// Relative allowance of one short chain of binary64 operations (conversion of an integer below 2^113,
// a logarithm faithful to an ulp, a few products and sums).
constexpr double unit = 16 * DBL_EPSILON;
double as_real(u128 x) {
    return std::ldexp(double(uint64_t(x >> 64)), 64) + double(uint64_t(x));
}
double down(double x, double allowance) {
    return x - allowance;
}
// The exact positive integer alpha k + beta.
u128 value_at(uint64_t alpha, int64_t beta, uint64_t k) {
    const u128 product = u128(alpha) * k;
    return beta < 0 ? product - (u128(uint64_t(-(beta + 1))) + 1) : product + u128(uint64_t(beta));
}
// log2 of the positive integer alpha k + beta with an absolute allowance.
LogBounds factor_log2(uint64_t alpha, int64_t beta, uint64_t k) {
    const double v = std::log2(as_real(value_at(alpha, beta, k)));
    const double e = unit * (std::fabs(v) + 1);
    return {down(v, e), v + e};
}
} // namespace
LogBounds operator+(LogBounds x, LogBounds y) noexcept {
    const double lower = x.lower + y.lower, upper = x.upper + y.upper;
    const double e = DBL_EPSILON * (std::fabs(lower) + std::fabs(upper));
    return {lower - e, upper + e};
}
LogBounds operator-(LogBounds x, LogBounds y) noexcept {
    const double lower = x.lower - y.upper, upper = x.upper - y.lower;
    const double e = DBL_EPSILON * (std::fabs(lower) + std::fabs(upper));
    return {lower - e, upper + e};
}
LogBounds scaled(LogBounds x, double nonnegative) noexcept {
    const double lower = x.lower * nonnegative, upper = x.upper * nonnegative;
    const double e = DBL_EPSILON * (std::fabs(lower) + std::fabs(upper));
    return {lower - e, upper + e};
}
LogBounds linear_log2_sum(uint64_t alpha, int64_t beta, uint64_t a, uint64_t b) noexcept {
    require(alpha >= 1 && a <= b && b <= (uint64_t(1) << 48), SBN3_FATAL_ARGUMENT, "log product range");
    if (a == b)
        return {};
    {
        const u128 first = u128(alpha) * a;
        const u128 magnitude = beta < 0 ? u128(uint64_t(-(beta + 1))) + 1 : 0;
        require(beta >= 0 ? first + u128(uint64_t(beta)) >= 1 : first > magnitude, SBN3_FATAL_ARGUMENT,
                "log product factor must be positive");
    }
    LogBounds sum{};
    uint64_t k = a;
    // z = k + beta/alpha = (alpha k + beta)/alpha, formed from the exact integer
    // numerator so a large negative beta cannot cancel against k.
    auto position = [&](uint64_t index) { return as_real(value_at(alpha, beta, index)) / double(alpha); };
    // Sum directly while z < 8 and for short ranges. (8.5: the comparison itself is rounded.)
    while (k < b && (b - k <= 16 || position(k) < 8.5)) {
        sum = sum + factor_log2(alpha, beta, k);
        ++k;
    }
    if (k == b)
        return sum;
    const double h = double(b - k), z = position(k);
    const double ratio = std::log1p(h / z), lz = std::log(z);
    const double terms[]{h * lz, (z + h - 0.5) * ratio, -h, -h / (12 * z * (z + h))};
    double g = 0, mass = 0;
    for (double t : terms) {
        g += t;
        mass += std::fabs(t);
    }
    // Analytic remainder (5), evaluation allowance of the four terms and their sum, and the
    // sensitivity to the rounding of z itself: |dG/dz| <= log1p(h/z) + h/z^2.
    const double analytic = (1 / (z * z * z) + 1 / ((z + h) * (z + h) * (z + h))) / 360;
    const double rounding = unit * mass + (ratio + h / (z * z)) * (4 * DBL_EPSILON * z);
    const double scale = h * std::log2(double(alpha));
    const double radius = (analytic + rounding) / ln2 * (1 + unit) + unit * (std::fabs(scale) + 1);
    const double value = scale + g / ln2;
    const double slack = radius + unit * std::fabs(value);
    return sum + LogBounds{value - slack, value + slack};
}
bool product_log2_sum(const FactorProduct &f, uint64_t a, uint64_t b, LogBounds &out) noexcept {
    out = {};
    if (f.degree || a > b)
        return false;
    if (a == b)
        return true;
    const double c = std::log2(double(f.constant_low) + std::ldexp(double(f.constant_high), 64));
    const double h = double(b - a);
    const double e = unit * (std::fabs(c) + 1) * h;
    out = {c * h - e, c * h + e};
    for (unsigned i = 0; i < f.count; ++i) {
        const auto &x = f.factor[i];
        if (!x.a)
            return false; // constants are folded by normalization
        const u128 first = u128(x.a) * a;
        if (x.b < 0 && first <= u128(uint64_t(-(x.b + 1))) + 1)
            return false;
        if (x.b >= 0 && !first && !x.b)
            return false;
        out = out + scaled(linear_log2_sum(x.a, x.b, a, b), double(x.power));
    }
    return true;
}
bool product_log2_at(const FactorProduct &f, uint64_t k, LogBounds &out) noexcept {
    return product_log2_sum(f, k, k + 1, out);
}
} // namespace sbn::v3::series
