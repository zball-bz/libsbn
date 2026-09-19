#pragma once
#include <stddef.h>
namespace sbn::v3::inverse_tuning {
// One normalized reciprocal, including all required preparation. Zen5 /
// Clang 21.1.8: local u52 recurrence + exact correction / public Newton us:
// 513 limbs 20/129, 1023 50/143, 2049 147/199, 3073 272/270,
// 4097 421/292. Shared by exact division and radix scaling powers.
// Both routes satisfy the public <3-ulp inverse contract; the local route
// additionally returns the exact all-ones numerator quotient.
// Evidence: experiments/results/codex_divrem_radix_2026-09-19/local-reciprocal/.
inline constexpr size_t basecase_limbs = 3072;
inline double local_cost(size_t n) noexcept {
    // 17..32 use word division. The local Toom/Newton envelope follows
    // n^1.5 here (measured 129..3073); no operand-length lookup table.
    return n<=32 ? 100.+.33*double(n)*double(n) : 100.+1.7*double(n)*__builtin_sqrt(double(n));
}
}
