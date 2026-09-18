#pragma once
#include "series/formula_def.hpp"
// Logarithmic product estimator of the series planner (docs/series-precision-planner-math-2026-09-17.md §2).
//
//   sum_{k=a}^{b-1} log2(alpha k + beta)
//     = h log2(alpha) + [ln Gamma(z + h) - ln Gamma(z)] / ln 2,     h = b - a,  z = a + beta/alpha,
//   ln Gamma(z + h) - ln Gamma(z) = G(z, h) + E,
//   G(z, h) = h ln z + (z + h - 1/2) log1p(h/z) - h - h / (12 z (z + h)),
//   |E| <= 1/(360 z^3) + 1/(360 (z + h)^3)                       (z > 0; one Stirling correction).
// Front factors are summed directly until z >= 8, short ranges entirely, so the analytic radius stays
// below 1.6e-5 bit per linear factor. Every result is an interval [lower, upper] that contains the exact
// real value: the analytic remainder plus an explicit allowance for the floating-point evaluation.
// Evaluation is in binary64: its allowance (a few 1e-15 of the summed magnitudes, below 1e-2 bit even for
// 2^28-limb requests) is far inside what the consumers need (bit lengths and floors of bit counts), and
// planning calls this thousands of times per pass, where extended-precision logarithms cost five times more.
//
// These are analysis facts. Capacity may use `upper`, contribution certificates `lower`; only
// scheduling may consume the midpoint. A difference of two intervals must be formed as
// [L_b - U_a, U_b - L_a], never as a difference of two bounds of the same kind.
namespace sbn::v3::series {
struct LogBounds {
    double lower = 0, upper = 0;
    double point() const noexcept { return (lower + upper) / 2; }
    double radius() const noexcept { return (upper - lower) / 2; }
};
// Interval sums widen by the rounding of the additions themselves.
LogBounds operator+(LogBounds x, LogBounds y) noexcept;
LogBounds operator-(LogBounds x, LogBounds y) noexcept;
LogBounds scaled(LogBounds x, double nonnegative) noexcept;
// sum_{k in [a,b)} log2(alpha k + beta). Requires alpha >= 1 and alpha a + beta >= 1 (every factor of the
// range is a positive integer); a <= b < 2^48 + 1. An empty range is [0, 0].
LogBounds linear_log2_sum(uint64_t alpha, int64_t beta, uint64_t a, uint64_t b) noexcept;
// sum_{k in [a,b)} log2 |c prod_j (a_j k + b_j)^{m_j}| for a normalized factor product without a
// polynomial multiplier. Returns false when some factor is not positive on the whole range (the caller
// then uses its integer envelopes).
bool product_log2_sum(const FactorProduct &f, uint64_t a, uint64_t b, LogBounds &out) noexcept;
// log2 of the factor product at one index under the same conditions.
bool product_log2_at(const FactorProduct &f, uint64_t k, LogBounds &out) noexcept;
} // namespace sbn::v3::series
