#pragma once
#include "sbn3/newton.h"
#include "sbn3/value.h"
namespace sbn::v3::series {
struct ScaledMagnitude {
    sbn3_int_view value{};
    int64_t exponent2 = 0;
};
// floor(magnitude * 2^shift), count little-endian words, including virtual zeros.
// Exact out==in.data alias is supported; partial overlaps are not supported.
void scaled_slice(uint64_t *, size_t, sbn3_int_view, int64_t shift) noexcept;
// Optional parallel execution uses an idle, pre-existing team.
void scaled_slice(uint64_t *, size_t, sbn3_int_view, int64_t shift, sbn3_team *team) noexcept;
// |numerator/denominator| < 2^63, denominator positive. Normalize to a
// prequeried DIVIDE precision fractional+2, dividing the numerator by B.
// After Newton's <3-ulp result, drop one low word. Absolute result error is
// <2 units at scale B^fractional, including both input truncations.
void ratio_inputs(ScaledMagnitude numerator, ScaledMagnitude denominator, size_t fractional, uint64_t *A,
                  uint64_t *D, sbn3_team *team=nullptr) noexcept;
// Borrow the normalized quotient window after its width certificate.
// Lifetime and limb alignment are inherited from the quotient buffer.
sbn3_const_limbs ratio_result(const uint64_t *quotient, size_t fractional) noexcept;
// allow_lower_boundary is only for a known-nonnegative result whose retained
// words are zero. The upper boundary must still be separated from the error.
void guard_separated(const uint64_t *, size_t dropped_words, uint64_t error_units,
                     bool allow_lower_boundary = false) noexcept;
} // namespace sbn::v3::series
