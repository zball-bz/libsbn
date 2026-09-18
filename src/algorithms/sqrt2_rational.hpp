#pragma once
#include "sbn3/product.h"
namespace sbn::v3 {
// Pure integer sizing from a certified interval for log2(1+sqrt(2)).
size_t rational_sqrt2_capacity(unsigned iteration) noexcept;
unsigned rational_sqrt2_iterations(size_t fractional_limbs) noexcept;
struct RationalSqrt2Step {
    size_t input_limbs, output_limbs;
    sbn3_mul_binding *multiply, *square;
    sbn3_spectrum *p; // optional; short backends need no spectrum object
};
struct RationalSqrt2Program {
    unsigned last_iteration;
    const RationalSqrt2Step *steps; // iterations 1->2 ... last-1->last
    uint64_t *values[3];            // each rational_sqrt2_capacity(last)+2 limbs
};
struct RationalSqrt2Value {
    const uint64_t *p, *q;
    size_t limbs;
};
RationalSqrt2Value rational_sqrt2(const RationalSqrt2Program &) noexcept;
// Exact shared shift, making D a normalized n-limb integer. A and D each
// have n+1 limbs, with D[n]==0 and A[n]<=1. Their ratio remains p/q.
void normalize_rational_sqrt2(RationalSqrt2Value, size_t n, uint64_t *A, uint64_t *D) noexcept;
} // namespace sbn::v3
