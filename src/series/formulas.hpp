#pragma once
#include "sbn3/series.h"

namespace sbn::v3::series {
// Consumers of the finite-series scheduler, not branches in its executor.
// Exact BSR reference forms; partial-sum precision blocks are a separate layer.
enum class FormulaKind { Euler, Chudnovsky, BinaryLog };
struct Formula {
    FormulaKind kind;
    unsigned radix_bits = 1; // BinaryLog sum 2^(-r*k)/k; r=1 gives log(2).
    static constexpr uint64_t max_index = (uint64_t(1) << 48) - 1;
    sbn3_series_recipe recipe() const noexcept;
    sbn3_query_result bounds(sbn3_series_range, uint64_t max_terms, unsigned need,
                             sbn3_series_shape &, bool normalized = false) const noexcept;
    double work(sbn3_series_range) const noexcept;
    // Chudnovsky tree-only smooth selector and conservative serial envelope.
    // work() deliberately retains the existing PSR partition policy.
    double split_mass(sbn3_series_range) const noexcept;
    uint64_t split_point(sbn3_series_range, double fraction = .5) const noexcept;
    sbn3_query_result serial_envelope(sbn3_series_range, unsigned depth, unsigned need,
                                      uint64_t &max_terms, sbn3_series_shape &, bool normalized = false) const noexcept;
    // k>=1 for Euler/BinaryLog; k>=0 for Chudnovsky. Each supplied leaf
    // magnitude has capacity >=5, caller owned. No multi-limb division/heap.
    void leaf(uint64_t k, unsigned need, sbn3_series_values &) const noexcept;
    void euler_batch(sbn3_series_range, unsigned need, sbn3_series_values &) const noexcept;
};
// Integer-only bound: sum bit_length(k), a<=k<b, 1<=a<b<=2^48.
// O(log b) buckets, including the discontinuity exactly at powers of two.
__uint128_t integer_log_sum(uint64_t a, uint64_t b) noexcept;
struct FactorialLogBounds { uint64_t lower, upper; };
// Integer bounds on log2(b!/a!), 0<=a<=b<=2^48. Stable log1p increment,
// trapezoid remainder and an explicit native-libm rounding allowance.
FactorialLogBounds factorial_log_bounds(uint64_t a, uint64_t b) noexcept;
} // namespace sbn::v3::series
